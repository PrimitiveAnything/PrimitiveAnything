from math import ceil
from pathlib import Path
import os
import re

from beartype.typing import Tuple
from einops import rearrange, repeat
from toolz import valmap
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import yaml

from .typing import typecheck


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def path_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path, default_path=None):
    path = path_exists(path)
    with open(path, mode='r') as fp:
        cfg_s = yaml.load(fp, Loader=yaml.FullLoader)

    if default_path is not None:
        default_path = path_exists(default_path)
        with open(default_path, mode='r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        # try current dir default
        default_path = path.parent / 'default.yml'
        if default_path.exists():
            with open(default_path, mode='r') as fp:
                cfg = yaml.load(fp, Loader=yaml.FullLoader)
        else:
            cfg = {}

    update_recursive(cfg, cfg_s)
    return cfg


def dump_yaml(cfg, path):
    with open(path, mode='w') as f:
        return yaml.safe_dump(cfg, f)


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def load_latest_checkpoint(checkpoint_dir):
    pattern = re.compile(rf".+\.ckpt\.(\d+)\.pt")
    max_epoch = -1
    latest_checkpoint = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            num_epoch = int(match.group(1))
            if num_epoch > max_epoch:
                max_epoch = num_epoch
                latest_checkpoint = checkpoint_dir / filename

    if not exists(latest_checkpoint):
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    checkpoint = torch.load(latest_checkpoint)
    return checkpoint, latest_checkpoint


def torch_to(inp, device, non_blocking=False):
    nb = non_blocking  # set to True when doing distributed jobs
    if isinstance(inp, torch.Tensor):
        return inp.to(device, non_blocking=nb)
    elif isinstance(inp, (list, tuple)):
        return type(inp)(map(lambda t: t.to(device, non_blocking=nb) if isinstance(t, torch.Tensor) else t, inp))
    elif isinstance(inp, dict):
        return valmap(lambda t: t.to(device, non_blocking=nb) if isinstance(t, torch.Tensor) else t, inp)
    else:
        raise NotImplementedError
    

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(x):
    return len(x) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

def masked_mean(tensor, mask, dim = -1, eps = 1e-5):
    if not exists(mask):
        return tensor.mean(dim = dim)

    mask = rearrange(mask, '... -> ... 1')
    tensor = tensor.masked_fill(~mask, 0.)

    total_el = mask.sum(dim = dim)
    num = tensor.sum(dim = dim)
    den = total_el.float().clamp(min = eps)
    mean = num / den
    mean = mean.masked_fill(total_el == 0, 0.)
    return mean

def cycle(dl):
    while True:
        for data in dl:
            yield data

def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]


# tensor helper functions

@typecheck
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@typecheck
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@typecheck
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.,
    kernel_size: int = 5
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * kernel_size))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)
    return rearrange(out, 'b c n -> b n c')

@typecheck
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob