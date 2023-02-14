import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import torch
import torch.nn as nn
from utils import helper as h
from utils import net


def to_torch(xs, device, dtype=torch.float32):
    return tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xs)


class Actor(nn.Module):
    def __init__(self, latent_dim, mlp_dims, action_shape):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(latent_dim, mlp_dims[0]),
                            nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._actor = net.mlp(mlp_dims[0], mlp_dims[1:], action_shape[0])
        self.apply(net.orthogonal_init) 

    def forward(self, obs, std):
        feature = self.trunk(obs)
        mu = self._actor(feature)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std)
    

class Critic(nn.Module):
    def __init__(self, latent_dim, mlp_dims, action_shape):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(latent_dim+action_shape[0], mlp_dims[0]),
                            nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._critic1 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self._critic2 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self.apply(net.orthogonal_init)

    def forward(self, z, a):
        feature = torch.cat([z, a], dim=-1)
        feature = self.trunk(feature)
        return self._critic1(feature), self._critic2(feature)


class Encoder(nn.Module):
    def __init__(self, obs_shape, mlp_dims, latent_dim):
        super().__init__()
        self._encoder = net.mlp(obs_shape[0], mlp_dims, latent_dim,)
        self.apply(net.orthogonal_init)

    def forward(self, obs):
        out = self._encoder(obs)
        return out


class LatentModel(nn.Module):
    def __init__(self, latent_dim, action_shape, mlp_dims,):
        super().__init__()
        self._dynamics = net.mlp(latent_dim+action_shape[0],mlp_dims, latent_dim)
        self._reward = net.mlp(latent_dim+action_shape[0], mlp_dims, 1)
        self.apply(net.orthogonal_init)

    def forward(self, z, action):
        """Perform one step forward rollout to predict the next latent state and reward."""
        assert z.ndim == action.ndim == 2 # [batch_dim, a/s_dim]

        x = torch.cat([z, action], dim=-1) # shape[B, xdim]
        next_z = self._dynamics(x)
        reward = self._reward(x)
        return next_z, reward


class TCRL(object):
    def __init__(self, obs_shape, action_shape, mlp_dims, latent_dim,
                lr, weight_decay=1e-6, tau=0.005, rho=0.9, gamma=0.99,
                nstep=3, horizon=5, state_coef=1.0, reward_coef=1.0, grad_clip_norm=10., 
                std_schedule="", std_clip=0.3, 
                device='cuda'):
        self.device = torch.device(device)
        
        # models
        self.encoder = Encoder(obs_shape, mlp_dims, latent_dim).to(self.device)
        self.encoder_tar = Encoder(obs_shape, mlp_dims, latent_dim).to(self.device) 

        self.trans= LatentModel(latent_dim, action_shape, mlp_dims,).to(self.device)

        self.actor = Actor(latent_dim, mlp_dims, action_shape).to(self.device)
        
        self.critic = Critic(latent_dim, mlp_dims, action_shape).to(self.device)
        self.critic_tar = Critic(latent_dim, mlp_dims, action_shape).to(self.device)
        
        # init optimizer
        self.enc_trans_optim = torch.optim.Adam(list(self.encoder.parameters()) + \
                                                list(self.trans.parameters()), lr=lr, weight_decay=weight_decay)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # assign variables
        # common
        self.counter = 0
        self.action_shape = action_shape
        self.tau = tau # EMA coef

        self.std_schedule = std_schedule
        self.std = h.linear_schedule(self.std_schedule, 0)
        self.std_clip = std_clip  

        # transition related 
        self.state_coef, self.reward_coef, self.horizon, self.rho, self.grad_clip_norm = state_coef, reward_coef, horizon, rho, grad_clip_norm
    
        # policy related
        self.gamma, self.nstep = gamma, nstep


    def save(self, fp):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'trans': self.trans.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, fp)

    def load(self, fp):
        d = torch.load(fp)
        self.encoder.load_state_dict(d['encoder'])
        self.trans.load_state_dict(d['trans'])
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])

    @torch.no_grad()
    def enc(self, obs):
        """ Only replace part of the states from the original observations to check which one have the highest impacts."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.encoder(obs)

    def _update_enc_trans(self, obs, action, next_obses, reward):

        self.enc_trans_optim.zero_grad(set_to_none=True)
        self.trans.train()

        state_loss, reward_loss = 0, 0
        
        z = self.encoder(obs)

        for t in range(self.horizon):
            next_z_pred, r_pred = self.trans(z, action[t])

            with torch.no_grad():
                next_obs = next_obses[t]
                next_z_tar = self.encoder_tar(next_obs)
                r_tar = reward[t]
                assert next_obs.ndim == r_tar.ndim == 2

            # Losses
            rho = (self.rho ** t)            
            state_loss += rho * torch.mean(h.cosine(next_z_pred, next_z_tar), dim=-1) 
            reward_loss += rho * torch.mean(h.mse(r_pred, r_tar), dim=-1)
            
            # don't forget this
            z = next_z_pred

        total_loss = (self.state_coef * state_loss.clamp(max=1e4) + \
                    self.reward_coef * reward_loss.clamp(max=1e4)).mean()

        total_loss.register_hook(lambda grad: grad * (1 / self.horizon))
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.trans.parameters()), self.grad_clip_norm, error_if_nonfinite=False)

        self.enc_trans_optim.step()

        self.trans.eval()
        return {
                'trans_loss': float(total_loss.mean().item()),
                'consistency_loss': float(state_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'trans_grad_norm': float(grad_norm),
                'z_mean': z.mean().item(), 'z_max':z.max().item(), 'z_min':z.min().item()
                }
   
    def _update_q(self, z, act, rew, discount, next_z):
        with torch.no_grad():
            action = self.actor(next_z, std=self.std).sample(clip=self.std_clip)

            td_target = rew + discount * \
                torch.min(*self.critic_tar(next_z, action)) 

        q1, q2 = self.critic(z, act)
        q_loss = torch.mean(h.mse(q1, td_target) + h.mse(q2, td_target))
        
        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_optim.step() 

        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}

    def _update_pi(self, z):
        a = self.actor(z, std=self.std).sample(clip=self.std_clip)
        Q = torch.min(*self.critic(z, a))
        pi_loss = -Q.mean()
        
        self.actor_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.actor_optim.step()

        return {'pi_loss':pi_loss.item()}
        
    def update(self, step, replay_iter, batch_size):
        self.std = h.linear_schedule(self.std_schedule, step) # linearly udpate std
        info = {'std': self.std}

        # obs, action, next_obses, reward = replay_buffer.sample(batch_size) 
        batch = next(replay_iter)
        obs, action, reward, discount, next_obses = to_torch(batch, self.device, dtype=torch.float32)
        # swap the batch and horizon dimension -> [H, B, _shape]
        action, reward, discount, next_obses = torch.swapaxes(action, 0, 1), torch.swapaxes(reward, 0, 1),\
                                             torch.swapaxes(discount, 0, 1), torch.swapaxes(next_obses, 0, 1)
        
        # update encoder and latent dynamics
        info.update(self._update_enc_trans(obs, action, next_obses, reward))
        
        # form n-step samples
        z0, zt = self.enc(obs), self.enc(next_obses[self.nstep-1])
        _rew, _discount = 0, 1
        for t in range(self.nstep):
            _rew += _discount * reward[t]
            _discount *= self.gamma
        # udpate policy and value functions
        info.update(self._update_q(z0, action[0], _rew, _discount, zt))
        info.update(self._update_pi(z0))
        
        # update target networks
        h.soft_update_params(self.encoder, self.encoder_tar, self.tau)
        h.soft_update_params(self.critic, self.critic_tar, self.tau)
             
        self.counter += 1

        return info 

    @torch.no_grad()
    def select_action(self, obs, eval_mode=False, t0=True):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        dist = self.actor(self.enc(obs), std=self.std)
        if eval_mode: 
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action[0]
