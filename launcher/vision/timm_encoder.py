from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm

# For timm to work without SSL verification
import os
os.environ['CURL_CA_BUNDLE'] = ''


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height: int, width: int, channel: int, num_features: int):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(height, width, channel, num_features))
        fan_in = float(height * width * channel)
        std = 1.0 / torch.sqrt(torch.tensor(fan_in))
        nn.init.normal_(self.kernel, mean=0.0, std=std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        no_batch_dim = len(x.shape) < 4
        if no_batch_dim:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        x = x.unsqueeze(-1) * self.kernel.unsqueeze(0)
        x = x.sum(dim=(1, 2))
        x = x.reshape(batch_size, -1)

        if no_batch_dim:
            x = x.squeeze(0)

        return x


class PreTrainedResNetEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        pooling_method: str = "spatial_learned_embeddings",
        num_spatial_blocks: int = 8,
        bottleneck_dim: Optional[int] = 256,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        image_size: Tuple[int, int] = (128, 128),
        dropout_rate: float = 0.1,
        shared_backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone
        
        # Use shared backbone if provided, otherwise create new one
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='',  # Remove global pooling to keep spatial dims
            )
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()
        
        # Get output channels by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size[0], image_size[1])
            dummy_output = self.backbone(dummy_input)
            # timm outputs [B, C, H, W]
            self.feature_channels = dummy_output.shape[1]
            self.feature_height = dummy_output.shape[2]
            self.feature_width = dummy_output.shape[3]
    

        if pooling_method == "spatial_learned_embeddings":
            self.pooling = SpatialLearnedEmbeddings(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
                num_features=num_spatial_blocks,
            )
            self.dropout = nn.Dropout(dropout_rate)
            pooled_dim = self.feature_channels * num_spatial_blocks
        elif pooling_method == "avg":
            self.pooling = None
            self.dropout = None
            pooled_dim = self.feature_channels
        elif pooling_method == "max":
            self.pooling = None
            self.dropout = None
            pooled_dim = self.feature_channels
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        # Bottleneck projection (independent for each encoder instance)
        if bottleneck_dim is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(pooled_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.Tanh(),
            )
            nn.init.xavier_uniform_(self.bottleneck[0].weight)
            nn.init.zeros_(self.bottleneck[0].bias)
            self.output_dim = bottleneck_dim
        else:
            self.bottleneck = None
            self.output_dim = pooled_dim
        
        # ImageNet normalization
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(
        self,
        observations: torch.Tensor
    ) -> torch.Tensor:
        x = observations

        # Handle [B, H, W, C] -> [B, C, H, W] （from gym to torch）
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and x.shape[-1] in [1, 3]:
            x = x.permute(2, 0, 1).unsqueeze(0)
        
        x = x.float() / 255.0
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        x = self.backbone(x)
        
        x = x.permute(0, 2, 3, 1)
        
        if self.pooling_method == "spatial_learned_embeddings":
            x = self.pooling(x)
            x = self.dropout(x)
        elif self.pooling_method == "avg":
            x = x.mean(dim=(1, 2))  # [B, C]
        elif self.pooling_method == "max":
            x = x.amax(dim=(1, 2))  # [B, C]
            
        # Apply bottleneck
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def create_encoder(
    encoder_type: str = "resnet18-pretrained",
    image_keys: Tuple[str, ...] = ("image",),
    image_size: Tuple[int, int] = (128, 128),
    pooling_method: str = "max",
    num_spatial_blocks: int = 8,
    bottleneck_dim: int = 256,
    shared_backbone: Optional[nn.Module] = None,
) -> dict:
    if encoder_type == "resnet18-pretrained":
        model_name = "resnet18"
        pretrained = True
        freeze_backbone = True
    elif encoder_type == "resnet18":
        model_name = "resnet18"
        pretrained = False
        freeze_backbone = False
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")
    
    encoders = {}
    for image_key in image_keys:
        encoders[image_key] = PreTrainedResNetEncoder(
            model_name=model_name,
            pooling_method=pooling_method,
            num_spatial_blocks=num_spatial_blocks,
            bottleneck_dim=bottleneck_dim,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            image_size=image_size,
            shared_backbone=shared_backbone,
        )
    
    return encoders