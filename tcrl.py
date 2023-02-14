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
                use_model="pass", mppi_kwargs="",
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
        
        self.mppi_kwargs = mppi_kwargs
        self.use_model = use_model


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


    def _update_trans(self, obs, action, next_obses, reward):

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

    def _update_q_gae(self, z):
        zs, acts, rs, qs = [z], [], [], []
        H = 5

        with torch.no_grad():
            for t in range(H):
                act = self.actor(z, self.std).sample(self.std_clip)
                acts.append(act)
                qs.append(torch.min(*self.critic_tar(z, act)))
                z, r = self.trans(z, act)
                zs.append(z)
                rs.append(r)
            
            # calculate td_target
            next_q = torch.min(*self.critic_tar(z, self.actor(z, self.std).sample(self.std_clip)))
            
            gae_lambda = 0.95
            lastgaelam = 0.

            td_targets = []
            for t in reversed(range(len(rs))):
                if t == len(rs) - 1: # the last timestep
                    next_values = next_q
                else:
                    next_values = qs[t]
                
                delta = rs[t] + self.gamma * next_values - qs[t]
                lastgaelam = delta + self.gamma * gae_lambda * lastgaelam
                
                td_targets.append(lastgaelam + qs[t])

            td_targets = list(reversed(td_targets))

        # calculate the td error
        q_loss = 0
        for t, td_target in enumerate(td_targets):
            q1, q2 = self.critic(zs[t], acts[t])
            q_loss = h.mse(q1, td_target) + h.mse(q2, td_target) 
        q_loss = torch.mean(q_loss)

        # H-step td
        # q1, q2 = self.critic(zs[0], acts[0])
        # q_loss = torch.mean(h.mse(q1, td_targets[0]) + h.mse(q2, td_targets[0]))

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.register_hook(lambda grad: grad * (1 / H))
        q_loss.backward()
        self.critic_optim.step() 
        
        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}

    def _update_q_mve(self, z):
        zs, acts, rs, qs = [z], [], [], []
        H = 5

        with torch.no_grad():
            for t in range(H):
                act = self.actor(z, self.std).sample(self.std_clip)
                acts.append(act)
                z, r = self.trans(z, act)
                zs.append(z)
                rs.append(r)
            
            # calculate td_target
            next_q = torch.min(*self.critic_tar(z, self.actor(z, self.std).sample(self.std_clip)))


            td_targets = []
            for t in reversed(range(len(rs))):
                next_q = rs[t] + self.gamma * next_q
                td_targets.append(next_q)
            td_targets = list(reversed(td_targets))

        # calculate the td error
        q_loss = 0
        for t, td_target in enumerate(td_targets):
            q1, q2 = self.critic(zs[t], acts[t])
            q_loss = h.mse(q1, td_target) + h.mse(q2, td_target) 
        q_loss = torch.mean(q_loss)
        # H-step td
        # q1, q2 = self.critic(zs[0], acts[0])
        # q_loss = torch.mean(h.mse(q1, td_targets[0]) + h.mse(q2, td_targets[0]))

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.register_hook(lambda grad: grad * (1 / H))
        q_loss.backward()
        self.critic_optim.step() 
        
        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}
    
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
    
    def _update_pi_bp(self, z): 
        # imagine a trajectory
        H = 5       
        Q = 0

        for t in range(H):
            act = self.actor(z, self.std).sample(self.std_clip)
            z, r = self.trans(z, act)
            Q += (self.gamma**t) * r

        # calculate td_target
        next_q = torch.min(*self.critic_tar(z, self.actor(z, self.std).sample(self.std_clip)))
        Q += (self.gamma**H) * next_q

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
        
        info.update(self._update_trans(obs, action, next_obses, reward))
        
        if self.use_model == "pass":
            # form n-step samples
            z0, zt = self.enc(obs), self.enc(next_obses[self.nstep-1])
            _rew, _discount = 0, 1
            for t in range(self.nstep):
                _rew += _discount * reward[t]
                _discount *= self.gamma
            info.update(self._update_q(z0, action[0], _rew, _discount, zt))
            info.update(self._update_pi(z0))
        elif self.use_model == "gae":
            z0 = self.enc(obs)
            info.update(self._update_q_gae(z0))
            info.update(self._update_pi(z0))
        elif self.use_model == "mve":
            z0 = self.enc(obs)
            info.update(self._update_q_mve(z0))
            info.update(self._update_pi(z0))
        elif self.use_model == "bp":
            # form n-step samples
            z0, zt = self.enc(obs), self.enc(next_obses[self.nstep-1])
            _rew, _discount = 0, 1
            for t in range(self.nstep):
                _rew += _discount * reward[t]
                _discount *= self.gamma
            info.update(self._update_q(z0, action[0], _rew, _discount, zt))
            info.update(self._update_pi_bp(z0))
        else:
            raise ValueError
            
        # update target networks
        h.soft_update_params(self.encoder, self.encoder_tar, self.tau)
        h.soft_update_params(self.critic, self.critic_tar, self.tau)
             
        self.counter += 1

        return info 

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.trans(z, actions[t])
            G += discount * reward
            discount *= self.gamma
        G += discount * torch.min(*self.critic(z, self.actor(z, self.std).sample(clip=self.std_clip)))
        return G

    @torch.no_grad()
    def select_action(self, obs, eval_mode=False, t0=True):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if self.mppi_kwargs is None:
            dist = self.actor(self.enc(obs), std=self.std)
            if eval_mode: 
                action = dist.mean
            else:
                action = dist.sample(clip=None)
            return action[0]

        # sample policy trajectories
        horizon = self.mppi_kwargs.get('plan_horizon')
        num_samples = self.mppi_kwargs.get('num_samples')
        num_pi_trajs = int(self.mppi_kwargs.get('mixture_coef') * num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.action_shape[0], device=self.device)
            z = self.encoder(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.actor(z, self.std).sample()
                z, _ = self.trans(z, pi_actions[t])

        # Initialize state and parameters
        z = self.encoder(obs).repeat(num_samples+num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.action_shape[0], device=self.device)
        std = 2*torch.ones(horizon, self.action_shape[0], device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.mppi_kwargs.get('iteration')):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, num_samples, self.action_shape[0], device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.mppi_kwargs.get('num_elites'), dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.mppi_kwargs.get('temperature')*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(0.1, 2)
            mean, std = self.mppi_kwargs.get('momentum') * mean + (1 - self.mppi_kwargs.get('momentum')) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.action_shape[0], device=std.device)
        return a

