import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=64,
    ):
        super().__init__()
        block_channel = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, block_channel[0], kernel_size=1),
            nn.BatchNorm1d(block_channel[0]),
            nn.ReLU(),
            nn.Conv1d(block_channel[0], block_channel[1], kernel_size=1),
            nn.BatchNorm1d(block_channel[1]),
            nn.ReLU(),
            nn.Conv1d(block_channel[1], block_channel[2], kernel_size=1),
            nn.BatchNorm1d(block_channel[2]),
            nn.ReLU(),
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(block_channel[-1], out_channels),
            nn.BatchNorm1d(out_channels)
        )

        self.output_dim = out_channels
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        x = self.final_projection(x)
        return x