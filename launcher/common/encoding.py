from typing import Dict, Iterable
import torch
import torch.nn as nn
from einops import rearrange


class EncodingWrapper(nn.Module):
    def __init__(
        self,
        encoder: Dict[str, nn.Module],
        use_proprio: bool = False,
        proprio_latent_dim: int = 64,
        enable_stacking: bool = False,
        state_dim=None,
        image_keys: Iterable[str] = ("image",),
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoder)
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = list(image_keys)
        self.state_only = len(self.encoders) == 0 or len(self.image_keys) == 0

        if not self.state_only:
            first_encoder = list(self.encoders.values())[0]
            self.encoder_output_dim = first_encoder.output_dim
            self.total_encoder_dim = self.encoder_output_dim * len(self.image_keys)
        else:
            self.encoder_output_dim = 0
            self.total_encoder_dim = 0

        if use_proprio or self.state_only:
            assert state_dim is not None, "state_dim required for proprio or state-only"
            self.proprio_encoder = nn.Linear(state_dim, self.proprio_latent_dim)
            nn.init.xavier_uniform_(self.proprio_encoder.weight)
            nn.init.zeros_(self.proprio_encoder.bias)
            self.proprio_layer_norm = nn.LayerNorm(self.proprio_latent_dim)
            self.output_dim = self.total_encoder_dim + proprio_latent_dim
        else:
            self.proprio_encoder = None
            self.proprio_layer_norm = None
            self.output_dim = self.total_encoder_dim

    def forward(
        self,
        observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        encoded = []

        for image_key in self.image_keys:
            if image_key not in self.encoders:
                continue
            image = observations[image_key]

            if self.enable_stacking and len(image.shape) == 5:
                image = rearrange(image, "B T H W C -> (B T) H W C")
            elif self.enable_stacking and len(image.shape) == 4 and "point" in image_key:
                image = rearrange(image, "B T N C -> (B T) N C")

            features = self.encoders[image_key](image)
            encoded.append(features)

        if encoded:
            encoded = torch.cat(encoded, dim=-1)
        else:
            encoded = None

        if self.proprio_encoder is not None:
            state = observations["state"]
            if self.enable_stacking:
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    if encoded is not None:
                        encoded = encoded.reshape(-1)
                elif len(state.shape) == 3:
                    state = rearrange(state, "B T C -> (B T) C")

            state = self.proprio_encoder(state)
            state = self.proprio_layer_norm(state)
            state = torch.tanh(state)

            if encoded is not None:
                encoded = torch.cat([encoded, state], dim=-1)
            else:
                encoded = state

        return encoded