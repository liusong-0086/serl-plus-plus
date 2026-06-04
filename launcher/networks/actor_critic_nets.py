from typing import Optional, Sequence, Dict, Union
import torch
import torch.nn as nn
from torch.distributions import Normal

from launcher.utils.torch_utils import orthogonal_init
from launcher.networks.positional_embedding import SinusoidalPosEmb
from launcher.utils.torch_utils import append_zero, append_dims


class TanhNormal:
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        
    def mode(self) -> torch.Tensor:
        return torch.tanh(self.loc)
    
    def rsample(self, sample_shape=torch.Size()):
        u = self.base_dist.rsample(sample_shape)
        return torch.tanh(u)
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, actions):
        u = torch.atanh(torch.clamp(actions, -1 + 1e-7, 1 - 1e-7))
        return (self.base_dist.log_prob(u) - torch.log(1 - actions.pow(2) + 1e-7)).sum(dim=-1)
    
    def sample_and_log_prob(self, sample_shape=torch.Size()):
        u = self.base_dist.rsample(sample_shape)
        action = torch.tanh(u)
        
        log_prob = self.base_dist.log_prob(u)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        
        return action, log_prob.sum(dim=-1)


class ValueCritic(nn.Module):
    def __init__(
        self, 
        network: nn.Module,
        init_final: Optional[float] = None
    ):
        super().__init__()
        self.network = network
        self.init_final = init_final
        
        if init_final is not None:
            self.output_layer = nn.Linear(network.out_dim, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.out_dim, 1)
            orthogonal_init()(self.output_layer.weight)
            
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.network(observations, train)
        value = self.output_layer(x)
        return value.squeeze(-1)


class Critic(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        init_final: Optional[float] = None
    ):
        super().__init__()
        self.network = network
        self.init_final = init_final
        
        if init_final is not None:
            self.output_layer = nn.Linear(network.out_dim, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.out_dim, 1)
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        x = self.network(inputs)
        value = self.output_layer(x)
        return value.squeeze(-1)
    
    def q_value_ensemble(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        if len(actions.shape) == 3:  # [B, num_actions, action_dim]
            batch_size, num_actions = actions.shape[:2]
            obs_expanded = observations.unsqueeze(1).expand(-1, num_actions, -1)
            obs_flat = obs_expanded.reshape(-1, observations.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            q_values = self(obs_flat, actions_flat, train)
            return q_values.reshape(batch_size, num_actions)
        else:
            return self(observations, actions, train)


class GraspCritic(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        init_final: Optional[float] = None,
        output_dim: int = 3
    ):
        super().__init__()
        self.network = network
        self.init_final = init_final
        self.output_dim = output_dim
        
        if init_final is not None:
            self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            orthogonal_init()(self.output_layer.weight)
            
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.network(observations)
        return self.output_layer(x)  # [batch_size, output_dim] 


class CriticEnsemble(nn.Module):
    def __init__(self, critics: Sequence[nn.Module]):
        super().__init__()
        self.critics = nn.ModuleList(critics)
        self.num_critics = len(critics)
    
    def forward(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        q_values = [critic(observations, actions, train) for critic in self.critics]
        return torch.stack(q_values, dim=0)
    
    def q_min(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        q_values = self(observations, actions, train)
        return q_values.min(dim=0)[0]
    
    def q_mean(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        q_values = self(observations, actions, train)
        return q_values.mean(dim=0) 


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        action_dim: int,
        std_parameterization: str = "exp",
        std_min: float = 1e-5,
        std_max: float = 10.0,
        tanh_squash_distribution: bool = False,
        zero_mean_init: bool = False,
        fixed_std: Optional[torch.tensor] = None
    ):
        super().__init__()
        self.network = network
        self.action_dim = action_dim
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std
        
        self.mean_layer = nn.Linear(network.out_dim, action_dim, bias=(not zero_mean_init))
        if zero_mean_init:
            nn.init.zeros_(self.mean_layer.weight)
        else:
            orthogonal_init()(self.mean_layer.weight)
        
        if fixed_std is None:
            self.std_layer = nn.Linear(network.out_dim, action_dim)
            orthogonal_init()(self.std_layer.weight)
            
    def forward(
        self, 
        observations: torch.Tensor,
        temperature: float = 1.0,
        train: bool = False,
        non_squash_distribution: bool = False
    ):
        features = self.network(observations)
        means = self.mean_layer(features)
        
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(features)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.functional.softplus(self.std_layer(features))
            elif self.std_parameterization == "uniform":
                log_stds = self.std_layer.bias
                stds = torch.exp(log_stds).expand_as(means)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert self.std_parameterization == "fixed"
            stds = self.fixed_std.to(means.device).expand_as(means)
            
        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature, device=stds.device))
        
        if torch.isnan(means).any():
            means = torch.nan_to_num(means, nan=0.0)
        if torch.isnan(stds).any():
            stds = torch.nan_to_num(stds, nan=self.std_min)
        
        if self.tanh_squash_distribution and not non_squash_distribution:
            return TanhNormal(means, stds)
        else:
            return Normal(means, stds)


class ConsistencyPolicy(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        action_dim: int,
        t_dim : int = 16,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
        steps: int = 40,
        sigma_data: float = 0.5
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.steps = steps
        self.sigmas = self.get_karra_sigmas()
        self.action_dim = action_dim

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim)
        )
        self.network = network
        self.final_layer = nn.Linear(network.out_dim, action_dim)

    def get_karra_sigmas(self):
        ramp = torch.linspace(0, 1, self.steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return append_zero(sigmas)

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = ((sigma - self.sigma_min) * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5)
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, x_t, sigmas, obs_enc):
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) 
                               for x in self.get_scalings_for_boundary_condition(sigmas)]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        t = self.time_emb(rescaled_t)
        outputs = self.network(torch.cat([c_in * x_t, t, obs_enc], dim=-1))
        denoised = self.final_layer(outputs)
        denoised = c_out * denoised + c_skip * x_t
        return denoised
    
    def forward(self, obs_enc):
        had_one_dim = obs_enc.ndim == 1
        if had_one_dim:
            obs_enc = obs_enc.unsqueeze(0)
        x_t = torch.randn((obs_enc.shape[0], self.action_dim), device=obs_enc.device) * self.sigma_max
        sigmas = self.sigmas[0].expand(obs_enc.shape[0]).to(obs_enc.device)
        denoised = self.denoise(x_t, sigmas, obs_enc)
        out = denoised.clip(-1, 1)
        if had_one_dim:
            out = out.squeeze(0)
        return out