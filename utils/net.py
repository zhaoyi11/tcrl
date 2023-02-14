from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .helper import soft_clamp

def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ELU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int): raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims)-1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i+1]), act_fn()]

    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)

class StochasticMLP(nn.Module):
    def __init__(self, in_dim, mlp_dims, out_dim, act_fn=nn.ELU, out_act=nn.Identity):
        super().__init__()
        self.bnn = mlp(in_dim, mlp_dims, out_dim * 2, act_fn, out_act)  # output inclues mean and log scale
        
        self.apply(orthogonal_init)
        
        self.register_parameter('max_logvar', nn.Parameter(torch.ones(out_dim) * 0.5))
        self.register_parameter('min_logvar', nn.Parameter(torch.ones(out_dim) * -10)) 
        

    def forward(self, x, return_sample=True, eval_mode=False):
        _output = self.bnn(x)
        mu, logvar = torch.chunk(_output, 2, dim=-1) # shape of mu: [num_net, B, s_dim]
        batch_size, _ = mu.shape
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        std = torch.sqrt(torch.exp(logvar)) 

        dist = torch.distributions.Normal(mu, std)

        if return_sample:
            if eval_mode: return mu
            else: 
                return dist.rsample()
        else:
            return dist 


###### Ensembles ######
class EnsembleLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_net=7):
        super().__init__()
        # register weight and bias. The reason to use it rather than nn.Linear is to expand the ensemble_size dimension
        self.register_parameter('weight', nn.Parameter(torch.zeros(num_net, in_dim, out_dim)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_net, 1, out_dim)))

    
    def forward(self, x):
        weight = self.weight
        bias = self.bias
        # The linear layer has shape (ensemble_size, in_features, out_features).
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias
        return x


def emlp(in_dim, mlp_dims:List[int], out_dim, num_net, act_fn=nn.ELU, out_act=nn.Identity):
    """ Returns an ensemble of MLP."""
    if isinstance(mlp_dims, int): raise ValueError("mlp dimensions should be list, but got int.")

    layers = [EnsembleLinear(in_dim, mlp_dims[0], num_net), act_fn()]
    for i in range(len(mlp_dims)-1):
        layers += [EnsembleLinear(mlp_dims[i], mlp_dims[i+1], num_net), act_fn()]
    layers += [EnsembleLinear(mlp_dims[-1], out_dim, num_net), out_act()]
    return nn.Sequential(*layers)    


class StochasticEnsemble(nn.Module):
    def __init__(self, in_dim, mlp_dims, out_dim, num_net, act_fn=nn.ELU, out_act=nn.Identity):
        super().__init__()
        self.bnn = emlp(in_dim, mlp_dims, out_dim * 2, num_net, act_fn, out_act)  # output inclues mean and log scale
        
        self.apply(orthogonal_init)

        self.register_parameter('max_logvar', nn.Parameter(torch.ones(out_dim) * 0.5))
        self.register_parameter('min_logvar', nn.Parameter(torch.ones(out_dim) * -10)) 


    def forward(self, x):
        _output = self.bnn(x)
        mu, logvar = torch.chunk(_output, 2, dim=-1) # shape of mu: [num_net, B, s_dim]
        num_net, batch_size, _ = mu.shape
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        std = torch.sqrt(torch.exp(logvar))
        
        dist = torch.distributions.Normal(mu, std)
        return dist

class DeterministicEnsemble(nn.Module):
    def __init__(self, in_dim, mlp_dims, out_dim, num_net, act_fn=nn.ELU, out_act=nn.Identity):
        super().__init__()
        self.bnn = emlp(in_dim, mlp_dims, out_dim, num_net, act_fn, out_act)  # output inclues mean and log scale
        
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.bnn(x)
  

def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear): 
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, EnsembleLinear):
        for w in m.weight.data:
            nn.init.orthogonal_(w)
        if m.bias is not None:
            for b in m.bias.data:
                nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)