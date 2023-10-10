import torch
import torch.nn as nn

from guided_diffusion.restorer.rrdb_net import RRDBNet
from guided_diffusion.restorer.utils import linear, timestep_embedding

class Restorer(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN and Real-ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. # noqa: E501
    Currently, it supports [x1/x2/x4] upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
        upscale_factor (int): Upsampling factor. Support x1, x2 and x4.
            Default: 4.
    """

    def __init__(self,
                in_channels=9,
                out_channels=3,
                mid_channels=64,
                num_blocks=23,
                growth_channels=32,
                upscale_factor=1):

        super().__init__()

        # generator backbone
        self.generator = RRDBNet(
                            in_channels, 
                            out_channels, 
                            mid_channels, 
                            num_blocks, 
                            growth_channels, 
                            upscale_factor
        )

        # time embedding
        time_embed_dim = 3
        self.time_embed = nn.Sequential(
            linear(100, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

    def forward(self,
                x_t,
                y_t=None,
                t=None):
        """Forward function.

        Args:
            x_t (Tensor): Intermediate Tensor with shape (n, c, h, w).
            y_t (Tensor): LQ Tensor with shape (n, c, h, w).
            t (Tensor): Current timestep.

        Returns:
            Tensor: Output restored image with shape (n, c, h, w).
        """
        _model = self.generator
        
        t_emb = self.time_embed(timestep_embedding(t, 100)) # (B, 100) --> (B, 3)
        while len(t_emb.shape) < len(x_t.shape):
            t_emb = t_emb[..., None]
        t_emb = t_emb.expand(x_t.shape)

        x_t = torch.cat([x_t, y_t], dim=1)    # concat degraded lq
        x_t = torch.cat([x_t, t_emb], dim=1)  # concat time embeddings

        output = _model(x_t)

        return output