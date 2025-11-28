from __future__ import annotations

from functools import partial
from math import ceil
import os

from accelerate.utils import DistributedDataParallelKwargs
from beartype.typing import Tuple, Callable, List

from einops import rearrange, repeat, reduce, pack
from gateloop_transformer import SimpleGateLoopLayer
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import euler_angles_to_matrix
from x_transformers import Decoder
from x_transformers.x_transformers import LayerIntermediates
from x_transformers.autoregressive_wrapper import eval_decorator

from .michelangelo import ShapeConditioner as ShapeConditioner_miche
from .utils import (
    discretize,
    undiscretize,
    set_module_requires_grad_,
    default,
    exists,
    safe_cat,
    identity,
    is_tensor_empty,
)
from .utils.typing import Float, Int, Bool, typecheck


# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)
SHAPE_CODE = {
    'CubeBevel': 0,
    'SphereSharp': 1,
    'CylinderSharp': 2,
}
BS_NAME = {
    0: 'CubeBevel',
    1: 'SphereSharp',
    2: 'CylinderSharp',
}

# FiLM block

class FiLM(Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_gamma = nn.Linear(dim, dim_out, bias = False)
        self.to_beta = nn.Linear(dim, dim_out)

        self.gamma_mult = nn.Parameter(torch.zeros(1,))
        self.beta_mult = nn.Parameter(torch.zeros(1,))

    def forward(self, x, cond):
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = tuple(rearrange(t, 'b d -> b 1 d') for t in (gamma, beta))

        # for initializing to identity

        gamma = (1 + self.gamma_mult * gamma.tanh())
        beta = beta.tanh() * self.beta_mult

        # classic film

        return x * gamma + beta

# gateloop layers

class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    ):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim, use_heinsen = use_heinsen)
            self.gateloops.append(gateloop)

    def forward(
        self,
        x,
        cache = None
    ):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        return x, new_caches


def top_k_2(logits, frac_num_tokens=0.1, k=None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


def soft_argmax(labels):
    indices = torch.arange(labels.size(-1), dtype=labels.dtype, device=labels.device)
    soft_argmax = torch.sum(labels * indices, dim=-1)
    return soft_argmax


class PrimitiveTransformerDiscrete(Module, PyTorchModelHubMixin):
    @typecheck
    def __init__(
        self,
        *,
        num_discrete_scale = 128,
        continuous_range_scale: List[float, float] = [0, 1],
        dim_scale_embed = 64,
        num_discrete_rotation = 180,
        continuous_range_rotation: List[float, float] = [-180, 180],
        dim_rotation_embed = 64,
        num_discrete_translation = 128,
        continuous_range_translation: List[float, float] = [-1, 1],
        dim_translation_embed = 64,
        num_type = 3,
        dim_type_embed = 64,
        embed_order = 'ctrs',
        bin_smooth_blur_sigma = 0.4,
        dim: int | Tuple[int, int] = 512,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            attn_num_mem_kv = 4
        ),
        max_primitive_len = 144,
        dropout = 0.,
        coarse_pre_gateloop_depth = 2,
        coarse_post_gateloop_depth = 0,
        coarse_adaptive_rmsnorm = False,
        gateloop_use_heinsen = False,
        pad_id = -1,
        num_sos_tokens = None,
        condition_on_shape = True,
        shape_cond_with_cross_attn = False,
        shape_cond_with_film = False,
        shape_cond_with_cat = False,
        shape_condition_model_type = 'michelangelo',
        shape_condition_len = 1,
        shape_condition_dim = None,
        cross_attn_num_mem_kv = 4, # needed for preventing nan when dropping out shape condition
        loss_weight: dict = dict(
            eos = 1.0,
            type = 1.0,
            scale = 1.0,
            rotation = 1.0,
            translation = 1.0,
            reconstruction = 1.0,
            scale_huber = 1.0,
            rotation_huber = 1.0,
            translation_huber = 1.0,
        ),
        bs_pc_dir=None,
    ):
        super().__init__()

        # feature embedding
        self.num_discrete_scale = num_discrete_scale
        self.continuous_range_scale = continuous_range_scale
        self.discretize_scale = partial(discretize, num_discrete=num_discrete_scale, continuous_range=continuous_range_scale)
        self.undiscretize_scale = partial(undiscretize, num_discrete=num_discrete_scale, continuous_range=continuous_range_scale)
        self.scale_embed = nn.Embedding(num_discrete_scale, dim_scale_embed)

        self.num_discrete_rotation = num_discrete_rotation
        self.continuous_range_rotation = continuous_range_rotation
        self.discretize_rotation = partial(discretize, num_discrete=num_discrete_rotation, continuous_range=continuous_range_rotation)
        self.undiscretize_rotation = partial(undiscretize, num_discrete=num_discrete_rotation, continuous_range=continuous_range_rotation)
        self.rotation_embed = nn.Embedding(num_discrete_rotation, dim_rotation_embed)

        self.num_discrete_translation = num_discrete_translation
        self.continuous_range_translation = continuous_range_translation
        self.discretize_translation = partial(discretize, num_discrete=num_discrete_translation, continuous_range=continuous_range_translation)
        self.undiscretize_translation = partial(undiscretize, num_discrete=num_discrete_translation, continuous_range=continuous_range_translation)
        self.translation_embed = nn.Embedding(num_discrete_translation, dim_translation_embed)

        self.num_type = num_type
        self.type_embed = nn.Embedding(num_type, dim_type_embed)

        self.embed_order = embed_order
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

        # initial dimension
        
        self.dim = dim
        init_dim = 3 * (dim_scale_embed + dim_rotation_embed + dim_translation_embed) + dim_type_embed
        
        # project into model dimension
        self.project_in = nn.Linear(init_dim, dim)

        num_sos_tokens = default(num_sos_tokens, 1 if not condition_on_shape or not shape_cond_with_film else 4)
        assert num_sos_tokens > 0

        self.num_sos_tokens = num_sos_tokens
        self.sos_token = nn.Parameter(torch.randn(num_sos_tokens, dim))

        # the transformer eos token
        self.eos_token = nn.Parameter(torch.randn(1, dim))

        self.emb_layernorm = nn.LayerNorm(dim)
        self.max_seq_len = max_primitive_len

        # shape condition

        self.condition_on_shape = condition_on_shape
        self.shape_cond_with_cross_attn = False
        self.shape_cond_with_cat = False
        self.shape_condition_model_type = ''
        self.conditioner = None
        dim_shape = None

        if condition_on_shape:
            assert shape_cond_with_cross_attn or shape_cond_with_film or shape_cond_with_cat
            self.shape_cond_with_cross_attn = shape_cond_with_cross_attn
            self.shape_cond_with_cat = shape_cond_with_cat
            self.shape_condition_model_type = shape_condition_model_type
            if 'michelangelo' in shape_condition_model_type:
                self.conditioner = ShapeConditioner_miche(dim_latent=shape_condition_dim)
                self.to_cond_dim = nn.Linear(self.conditioner.dim_model_out * 2, self.conditioner.dim_latent)
                self.to_cond_dim_head = nn.Linear(self.conditioner.dim_model_out, self.conditioner.dim_latent)
            else:
                raise ValueError(f'unknown shape_condition_model_type {self.shape_condition_model_type}')

            dim_shape = self.conditioner.dim_latent
            set_module_requires_grad_(self.conditioner, False)

            self.shape_coarse_film_cond = FiLM(dim_shape, dim) if shape_cond_with_film else identity

        self.coarse_gateloop_block = GateLoopBlock(dim, depth=coarse_pre_gateloop_depth, use_heinsen=gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None
        self.coarse_post_gateloop_block = GateLoopBlock(dim, depth=coarse_post_gateloop_depth, use_heinsen=gateloop_use_heinsen) if coarse_post_gateloop_depth > 0 else None
        self.coarse_adaptive_rmsnorm = coarse_adaptive_rmsnorm

        self.decoder = Decoder(
            dim=dim,
            depth=attn_depth,
            heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_flash=flash_attn,
            attn_dropout=dropout,
            ff_dropout=dropout,
            use_adaptive_rmsnorm=coarse_adaptive_rmsnorm,
            dim_condition=dim_shape,
            cross_attend=self.shape_cond_with_cross_attn,
            cross_attn_dim_context=dim_shape,
            cross_attn_num_mem_kv=cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # to logits
        self.to_eos_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.to_type_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_type)
        )
        self.to_translation_logits = nn.Sequential(
            nn.Linear(dim + dim_type_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 3 * num_discrete_translation)
        )
        self.to_rotation_logits = nn.Sequential(
            nn.Linear(dim + dim_type_embed + 3 * dim_translation_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 3 * num_discrete_rotation)
        )
        self.to_scale_logits = nn.Sequential(
            nn.Linear(dim + dim_type_embed + 3 * (dim_translation_embed + dim_rotation_embed), dim),
            nn.ReLU(),
            nn.Linear(dim, 3 * num_discrete_scale)
        )

        self.pad_id = pad_id

        bs_pc_map = {}
        for bs_name, type_code in SHAPE_CODE.items():
            pc = o3d.io.read_point_cloud(os.path.join(bs_pc_dir, f'SM_GR_BS_{bs_name}_001.ply'))
            bs_pc_map[type_code] = torch.from_numpy(np.asarray(pc.points)).float()
        bs_pc_list = []
        for i in range(len(bs_pc_map)):
            bs_pc_list.append(bs_pc_map[i])
        self.bs_pc = torch.stack(bs_pc_list, dim=0)

        self.rotation_matrix_align_coord = euler_angles_to_matrix(
                                            torch.Tensor([np.pi/2, 0, 0]), 'XYZ').unsqueeze(0).unsqueeze(0)

    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    @torch.no_grad()
    def embed_pc(self, pc: Tensor):
        if 'michelangelo' in self.shape_condition_model_type:
            pc_head, pc_embed = self.conditioner(shape=pc)
            pc_embed = torch.cat([self.to_cond_dim_head(pc_head), self.to_cond_dim(pc_embed)], dim=-2).detach()
        else:
            raise ValueError(f'unknown shape_condition_model_type {self.shape_condition_model_type}')

        return pc_embed

    @typecheck
    def recon_primitives(
        self,
        scale_logits: Float['b np 3 nd'],
        rotation_logits: Float['b np 3 nd'],
        translation_logits: Float['b np 3 nd'],
        type_logits: Int['b np nd'],
        primitive_mask: Bool['b np']
    ):
        recon_scale = self.undiscretize_scale(scale_logits.argmax(dim=-1))
        recon_scale = recon_scale.masked_fill(~primitive_mask.unsqueeze(-1), float('nan'))
        recon_rotation = self.undiscretize_rotation(rotation_logits.argmax(dim=-1))
        recon_rotation = recon_rotation.masked_fill(~primitive_mask.unsqueeze(-1), float('nan'))
        recon_translation = self.undiscretize_translation(translation_logits.argmax(dim=-1))
        recon_translation = recon_translation.masked_fill(~primitive_mask.unsqueeze(-1), float('nan'))
        recon_type_code = type_logits.argmax(dim=-1)
        recon_type_code = recon_type_code.masked_fill(~primitive_mask, -1)
        
        return {
            'scale': recon_scale,
            'rotation': recon_rotation,
            'translation': recon_translation,
            'type_code': recon_type_code
        }
    
    @typecheck
    def sample_primitives(
        self,
        scale: Float['b np 3 nd'],
        rotation: Float['b np 3 nd'],
        translation: Float['b np 3 nd'],
        type_code: Int['b np nd'],
        next_embed: Float['b 1 nd'],
        temperature: float = 1.,
        filter_logits_fn: Callable = top_k_2,
        filter_kwargs: dict = dict()
    ):
        def sample_func(logits):
            if logits.ndim == 4:
                enable_squeeze = True
                logits = logits.squeeze(1)
            else:
                enable_squeeze = False

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                sample = filtered_logits.argmax(dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)

                sample = torch.zeros((probs.shape[0], probs.shape[1]), dtype=torch.long, device=probs.device)
                for b_i in range(probs.shape[0]):
                    sample[b_i] = torch.multinomial(probs[b_i], 1).squeeze()

            if enable_squeeze:
                sample = sample.unsqueeze(1)

            return sample
        
        next_type_logits = self.to_type_logits(next_embed)
        next_type_code = sample_func(next_type_logits)
        type_code_new, _ = pack([type_code, next_type_code], 'b *')

        type_embed = self.type_embed(next_type_code)
        next_embed_packed, _ = pack([next_embed, type_embed], 'b np *')
        next_translation_logits = rearrange(self.to_translation_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_translation)
        next_discretize_translation = sample_func(next_translation_logits)
        next_translation = self.undiscretize_translation(next_discretize_translation)
        translation_new, _ = pack([translation, next_translation], 'b * nd')
        
        next_translation_embed = self.translation_embed(next_discretize_translation)
        next_embed_packed, _ = pack([next_embed_packed, next_translation_embed], 'b np *')
        next_rotation_logits = rearrange(self.to_rotation_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_rotation)
        next_discretize_rotation = sample_func(next_rotation_logits)
        next_rotation = self.undiscretize_rotation(next_discretize_rotation)
        rotation_new, _ = pack([rotation, next_rotation], 'b * nd')

        next_rotation_embed = self.rotation_embed(next_discretize_rotation)
        next_embed_packed, _ = pack([next_embed_packed, next_rotation_embed], 'b np *')
        next_scale_logits = rearrange(self.to_scale_logits(next_embed_packed), 'b np (c nd) -> b np c nd', nd=self.num_discrete_scale)
        next_discretize_scale = sample_func(next_scale_logits)
        next_scale = self.undiscretize_scale(next_discretize_scale)
        scale_new, _ = pack([scale, next_scale], 'b * nd')

        return (
            scale_new,
            rotation_new,
            translation_new,
            type_code_new
        )

    @eval_decorator
    @torch.no_grad()
    @typecheck
    def generate(
        self,
        batch_size: int | None = None,
        filter_logits_fn: Callable = top_k_2,
        filter_kwargs: dict = dict(),
        temperature: float = 1.,
        scale: Float['b np 3'] | None = None,
        rotation: Float['b np 3'] | None = None,
        translation: Float['b np 3'] | None = None,
        type_code: Int['b np'] | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        cache_kv = True,
        max_seq_len = None,
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(scale) and exists(rotation) and exists(translation) and exists(type_code):
            assert not exists(batch_size)
            assert scale.shape[1] == rotation.shape[1] == translation.shape[1] == type_code.shape[1]
            assert scale.shape[1] <= self.max_seq_len
            
            batch_size = scale.shape[0]

        if self.condition_on_shape:
            assert exists(pc) ^ exists(pc_embed), '`pc` or `pc_embed` must be passed in'
            if exists(pc):
                pc_embed = self.embed_pc(pc)

            batch_size = default(batch_size, pc_embed.shape[0])

        batch_size = default(batch_size, 1)

        scale = default(scale, torch.empty((batch_size, 0, 3), dtype=torch.float64, device=self.device))
        rotation = default(rotation, torch.empty((batch_size, 0, 3), dtype=torch.float64, device=self.device))
        translation = default(translation, torch.empty((batch_size, 0, 3), dtype=torch.float64, device=self.device))
        type_code = default(type_code, torch.empty((batch_size, 0), dtype=torch.int64, device=self.device))

        curr_length = scale.shape[1]

        cache = None
        eos_codes = None

        for i in tqdm(range(curr_length, max_seq_len)):
            can_eos = i != 0
            output = self.forward(
                scale=scale,
                rotation=rotation,
                translation=translation,
                type_code=type_code,
                pc_embed=pc_embed,
                return_loss=False,
                return_cache=cache_kv,
                append_eos=False,
                cache=cache
            )
            if cache_kv:
                next_embed, cache = output
            else:
                next_embed = output
            (
                scale,
                rotation,
                translation,
                type_code
            ) = self.sample_primitives(
                scale,
                rotation,
                translation,
                type_code,
                next_embed,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs
            )

            next_eos_logits = self.to_eos_logits(next_embed).squeeze(-1)
            next_eos_code = (F.sigmoid(next_eos_logits) > 0.5)
            eos_codes = safe_cat([eos_codes, next_eos_code], 1)
            if can_eos and eos_codes.any(dim=-1).all():
                break

        # mask out to padding anything after the first eos
        mask = eos_codes.float().cumsum(dim=-1) >= 1

        # concat cur_length to mask
        mask = torch.cat((torch.zeros((batch_size, curr_length), dtype=torch.bool, device=self.device), mask), dim=-1)
        type_code = type_code.masked_fill(mask, self.pad_id)
        scale = scale.masked_fill(mask.unsqueeze(-1), self.pad_id)
        rotation = rotation.masked_fill(mask.unsqueeze(-1), self.pad_id)
        translation = translation.masked_fill(mask.unsqueeze(-1), self.pad_id)

        recon_primitives = {
            'scale': scale,
            'rotation': rotation,
            'translation': translation,
            'type_code': type_code
        }
        primitive_mask = ~eos_codes

        return recon_primitives, primitive_mask


    @eval_decorator
    @torch.no_grad()
    @typecheck
    def generate_w_recon_loss(
        self,
        batch_size: int | None = None,
        filter_logits_fn: Callable = top_k_2,
        filter_kwargs: dict = dict(),
        temperature: float = 1.,
        scale: Float['b np 3'] | None = None,
        rotation: Float['b np 3'] | None = None,
        translation: Float['b np 3'] | None = None,
        type_code: Int['b np'] | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        cache_kv = True,
        max_seq_len = None,
        single_directional = True,
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(scale) and exists(rotation) and exists(translation) and exists(type_code):
            assert not exists(batch_size)
            assert scale.shape[1] == rotation.shape[1] == translation.shape[1] == type_code.shape[1]
            assert scale.shape[1] <= self.max_seq_len
            
            batch_size = scale.shape[0]

        if self.condition_on_shape:
            assert exists(pc) ^ exists(pc_embed), '`pc` or `pc_embed` must be passed in'
            if exists(pc):
                pc_embed = self.embed_pc(pc)

            batch_size = default(batch_size, pc_embed.shape[0])

        batch_size = default(batch_size, 1)
        assert batch_size == 1 # TODO: support any batch size

        scale = default(scale, torch.empty((batch_size, 0, 3), dtype=torch.float32, device=self.device))
        rotation = default(rotation, torch.empty((batch_size, 0, 3), dtype=torch.float32, device=self.device))
        translation = default(translation, torch.empty((batch_size, 0, 3), dtype=torch.float32, device=self.device))
        type_code = default(type_code, torch.empty((batch_size, 0), dtype=torch.int64, device=self.device))

        curr_length = scale.shape[1]

        cache = None
        eos_codes = None
        last_recon_loss = 1
        for i in tqdm(range(curr_length, max_seq_len)):
            can_eos = i != 0
            output = self.forward(
                scale=scale,
                rotation=rotation,
                translation=translation,
                type_code=type_code,
                pc_embed=pc_embed,
                return_loss=False,
                return_cache=cache_kv,
                append_eos=False,
                cache=cache
            )
            if cache_kv:
                next_embed, cache = output
            else:
                next_embed = output
            (
                scale_new,
                rotation_new,
                translation_new,
                type_code_new
            ) = self.sample_primitives(
                scale,
                rotation,
                translation,
                type_code,
                next_embed,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs
            )

            next_eos_logits = self.to_eos_logits(next_embed).squeeze(-1)
            next_eos_code = (F.sigmoid(next_eos_logits) > 0.5)
            eos_codes = safe_cat([eos_codes, next_eos_code], 1)
            if can_eos and eos_codes.any(dim=-1).all():
                scale, rotation, translation, type_code = (
                    scale_new, rotation_new, translation_new, type_code_new)
                break

            recon_loss = self.compute_chamfer_distance(scale_new, rotation_new, translation_new, type_code_new, ~eos_codes, pc, single_directional)
            if recon_loss < last_recon_loss:
                last_recon_loss = recon_loss
                scale, rotation, translation, type_code = (
                    scale_new, rotation_new, translation_new, type_code_new)
            else:
                best_recon_loss = recon_loss
                best_primitives = dict(
                    scale=scale_new, rotation=rotation_new, translation=translation_new, type_code=type_code_new)
                success_flag = False
                print(f'last_recon_loss:{last_recon_loss}, recon_loss:{recon_loss} -> to find better primitive')
                for try_i in range(5):
                    (
                        scale_new,
                        rotation_new,
                        translation_new,
                        type_code_new
                    ) = self.sample_primitives(
                        scale,
                        rotation,
                        translation,
                        type_code,
                        next_embed,
                        temperature=1.0,
                        filter_logits_fn=filter_logits_fn,
                        filter_kwargs=filter_kwargs
                    )
                    recon_loss = self.compute_chamfer_distance(scale_new, rotation_new, translation_new, type_code_new, ~eos_codes, pc)
                    print(f'[try_{try_i}] last_recon_loss:{last_recon_loss}, best_recon_loss:{best_recon_loss}, cur_recon_loss:{recon_loss}')
                    if recon_loss < last_recon_loss:
                        last_recon_loss = recon_loss
                        scale, rotation, translation, type_code = (
                            scale_new, rotation_new, translation_new, type_code_new)
                        success_flag = True
                        break
                    else:
                        if recon_loss < best_recon_loss:
                            best_recon_loss = recon_loss
                            best_primitives = dict(
                                scale=scale_new, rotation=rotation_new, translation=translation_new, type_code=type_code_new)

                if not success_flag:
                    last_recon_loss = best_recon_loss
                    scale, rotation, translation, type_code = (
                        best_primitives['scale'], best_primitives['rotation'], best_primitives['translation'], best_primitives['type_code'])
                print(f'new_last_recon_loss:{last_recon_loss}')

        # mask out to padding anything after the first eos
        mask = eos_codes.float().cumsum(dim=-1) >= 1
        type_code = type_code.masked_fill(mask, self.pad_id)
        scale = scale.masked_fill(mask.unsqueeze(-1), self.pad_id)
        rotation = rotation.masked_fill(mask.unsqueeze(-1), self.pad_id)
        translation = translation.masked_fill(mask.unsqueeze(-1), self.pad_id)

        recon_primitives = {
            'scale': scale,
            'rotation': rotation,
            'translation': translation,
            'type_code': type_code
        }
        primitive_mask = ~eos_codes

        return recon_primitives, primitive_mask


    @typecheck
    def encode(
        self,
        *,
        scale: Float['b np 3'],
        rotation: Float['b np 3'],
        translation: Float['b np 3'],
        type_code: Int['b np'],
        primitive_mask: Bool['b np'],
        return_primitives = False
    ):
        """
        einops:
        b - batch
        np - number of primitives
        c - coordinates (3)
        d - embed dim
        """
    
        # compute feature embedding
        discretize_scale = self.discretize_scale(scale)
        scale_embed = self.scale_embed(discretize_scale)
        scale_embed = rearrange(scale_embed, 'b np c d -> b np (c d)')

        discretize_rotation = self.discretize_rotation(rotation)
        rotation_embed = self.rotation_embed(discretize_rotation)
        rotation_embed = rearrange(rotation_embed, 'b np c d -> b np (c d)')

        discretize_translation = self.discretize_translation(translation)
        translation_embed = self.translation_embed(discretize_translation)
        translation_embed = rearrange(translation_embed, 'b np c d -> b np (c d)')

        type_embed = self.type_embed(type_code.masked_fill(~primitive_mask, 0))

        # combine all features and project into model dimension
        if self.embed_order == 'srtc':
            primitive_embed, _ = pack([scale_embed, rotation_embed, translation_embed, type_embed], 'b np *')
        else:
            primitive_embed, _ = pack([type_embed, translation_embed, rotation_embed, scale_embed], 'b np *')

        primitive_embed = self.project_in(primitive_embed)
        primitive_embed = primitive_embed.masked_fill(~primitive_mask.unsqueeze(-1), 0.)

        if not return_primitives:
            return primitive_embed
        
        primitive_embed_unpacked = {
            'scale': scale_embed,
            'rotation': rotation_embed,
            'translation': translation_embed,
            'type_code': type_embed
        }

        primitives_gt = {
            'scale': discretize_scale,
            'rotation': discretize_rotation,
            'translation': discretize_translation,
            'type_code': type_code
        }
        
        return primitive_embed, primitive_embed_unpacked, primitives_gt

    @typecheck
    def compute_chamfer_distance(
        self,
        scale_pred: Float['b np 3'],
        rotation_pred: Float['b np 3'],
        translation_pred: Float['b np 3'],
        type_pred: Int['b np'],
        primitive_mask: Bool['b np'],
        pc: Tensor, # b, num_points, c
        single_directional = True
    ):
        scale_pred = scale_pred.float()
        rotation_pred = rotation_pred.float()
        translation_pred = translation_pred.float()

        pc_pred = apply_transformation(self.bs_pc.to(type_pred.device)[type_pred], scale_pred, torch.deg2rad(rotation_pred), translation_pred)
        pc_pred = torch.matmul(pc_pred, self.rotation_matrix_align_coord.to(type_pred.device))
        pc_pred_flat = rearrange(pc_pred, 'b np p c -> b (np p) c')
        pc_pred_sampled = random_sample_pc(pc_pred_flat, primitive_mask.sum(dim=-1, keepdim=True), n_points=self.bs_pc.shape[1])

        if single_directional:
            recon_loss, _ = chamfer_distance(pc[:, :, :3].float(), pc_pred_sampled.float(), single_directional=True) # single directional
        else:
            recon_loss, _ = chamfer_distance(pc_pred_sampled.float(), pc[:, :, :3].float())

        return recon_loss

    def forward(
        self,
        *,
        scale: Float['b np 3'],
        rotation: Float['b np 3'],
        translation: Float['b np 3'],
        type_code: Int['b np'],
        loss_reduction: str = 'mean',
        return_cache = False,
        append_eos = True,
        cache: LayerIntermediates | None = None,
        pc: Tensor | None = None,
        pc_embed: Tensor | None = None,
        **kwargs
    ):

        primitive_mask = reduce(scale != self.pad_id, 'b np 3 -> b np', 'all')

        if scale.shape[1] > 0:
            codes, primitives_embeds, primitives_gt = self.encode(
                scale=scale,
                rotation=rotation,
                translation=translation,
                type_code=type_code,
                primitive_mask=primitive_mask,
                return_primitives=True
            )
        else:
            codes = torch.empty((scale.shape[0], 0, self.dim), dtype=torch.float32, device=self.device)

        # handle shape conditions

        attn_context_kwargs = dict()

        if self.condition_on_shape:
            assert exists(pc) ^ exists(pc_embed), '`pc` or `pc_embed` must be passed in'

            if exists(pc):
                if 'michelangelo' in self.shape_condition_model_type:
                    pc_head, pc_embed = self.conditioner(shape=pc)
                    pc_embed = torch.cat([self.to_cond_dim_head(pc_head), self.to_cond_dim(pc_embed)], dim=-2)
                else:
                    raise ValueError(f'unknown shape_condition_model_type {self.shape_condition_model_type}')

            assert pc_embed.shape[0] == codes.shape[0], 'batch size of point cloud is not equal to the batch size of the primitive codes'

            pooled_pc_embed = pc_embed.mean(dim=1) # (b, shape_condition_dim)

            if self.shape_cond_with_cross_attn:
                attn_context_kwargs = dict(
                    context=pc_embed
                )

            if self.coarse_adaptive_rmsnorm:
                attn_context_kwargs.update(
                    condition=pooled_pc_embed
                )

        batch, seq_len, _ = codes.shape # (b, np, dim)
        device = codes.device
        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        if append_eos:
            assert exists(codes)
            code_lens = primitive_mask.sum(dim=-1)
            codes = pad_tensor(codes)

            batch_arange = torch.arange(batch, device=device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')
            codes[batch_arange, code_lens] = self.eos_token # (b, np+1, dim)

        primitive_codes = codes # (b, np, dim)

        primitive_codes_len = primitive_codes.shape[-2]

        (
            coarse_cache,
            coarse_gateloop_cache,
            coarse_post_gateloop_cache,
        ) = cache if exists(cache) else ((None,) * 3)

        if not exists(cache):
            sos = repeat(self.sos_token, 'n d -> b n d', b=batch)

            if self.shape_cond_with_cat:
                sos, _ = pack([pc_embed, sos], 'b * d')
            primitive_codes, packed_sos_shape = pack([sos, primitive_codes], 'b * d') # (b, n_sos+np, dim)

        # condition primitive codes with shape if needed
        if self.condition_on_shape:
            primitive_codes = self.shape_coarse_film_cond(primitive_codes, pooled_pc_embed)

        # attention on primitive codes (coarse)

        if exists(self.coarse_gateloop_block):
            primitive_codes, coarse_gateloop_cache = self.coarse_gateloop_block(primitive_codes, cache=coarse_gateloop_cache)

        attended_primitive_codes, coarse_cache = self.decoder( # (b, n_sos+np, dim) 
            primitive_codes,
            cache=coarse_cache,
            return_hiddens=True,
            **attn_context_kwargs
        )

        if exists(self.coarse_post_gateloop_block):
            primitive_codes, coarse_post_gateloop_cache = self.coarse_post_gateloop_block(primitive_codes, cache=coarse_post_gateloop_cache)

        embed = attended_primitive_codes[:, -(primitive_codes_len + 1):] # (b, np+1, dim)

        if not return_cache:
            return embed[:, -1:]

        next_cache = (
            coarse_cache,
            coarse_gateloop_cache,
            coarse_post_gateloop_cache
        )

        return embed[:, -1:], next_cache


def pad_tensor(tensor):
    if tensor.dim() == 3:
        bs, seq_len, dim = tensor.shape
        padding = torch.zeros((bs, 1, dim), dtype=tensor.dtype, device=tensor.device)
    elif tensor.dim() == 2:
        bs, seq_len = tensor.shape
        padding = torch.zeros((bs, 1), dtype=tensor.dtype, device=tensor.device)
    else:
        raise ValueError('Unsupported tensor shape: {}'.format(tensor.shape))
    
    return torch.cat([tensor, padding], dim=1)


def apply_transformation(pc, scale, rotation_vector, translation):
    bs, np, num_points, _ = pc.shape
    scaled_pc = pc * scale.unsqueeze(2)

    rotation_matrix = euler_angles_to_matrix(rotation_vector.view(-1, 3), 'XYZ').view(bs, np, 3, 3) # euler tmp
    rotated_pc = torch.einsum('bnij,bnpj->bnpi', rotation_matrix, scaled_pc)

    transformed_pc = rotated_pc + translation.unsqueeze(2)

    return transformed_pc


def random_sample_pc(pc, max_lens, n_points=10000):
    bs = max_lens.shape[0]
    max_len = max_lens.max().item() * n_points

    random_values = torch.rand(bs, max_len, device=max_lens.device)
    mask = torch.arange(max_len).expand(bs, max_len).to(max_lens.device) < (max_lens * n_points)
    masked_random_values = random_values * mask.float()
    _, indices = torch.topk(masked_random_values, n_points, dim=1)
    
    return pc[torch.arange(bs).unsqueeze(1), indices]