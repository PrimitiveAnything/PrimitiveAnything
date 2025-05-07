import os

from omegaconf import OmegaConf
import torch
from torch import nn

from .utils.misc import instantiate_from_config
from ..utils import default, exists


def load_model():
    model_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "shapevae-256.yaml"))
    # print(model_config)
    if hasattr(model_config, "model"):
        model_config = model_config.model
    ckpt_path = "./ckpt/shapevae-256.ckpt"

    model = instantiate_from_config(model_config, ckpt_path=ckpt_path)
    # model = model.cuda()
    model = model.eval()

    return model


class ShapeConditioner(nn.Module):
    def __init__(
        self,
        *,
        dim_latent = None
    ):
        super().__init__()
        self.model = load_model()

        self.dim_model_out = 768
        dim_latent = default(dim_latent, self.dim_model_out)
        self.dim_latent = dim_latent

    def forward(
        self,
        shape = None,
        shape_embed = None,
    ):
        assert exists(shape) ^ exists(shape_embed)

        if not exists(shape_embed):
            point_feature = self.model.encode_latents(shape)
            shape_latents = self.model.to_shape_latents(point_feature[:, 1:])
            shape_head = point_feature[:, 0:1]
            shape_embed = torch.cat([point_feature[:, 1:], shape_latents], dim=-1)
            # shape_embed = torch.cat([point_feature[:, 1:], shape_latents], dim=-2) # cat tmp
        return shape_head, shape_embed