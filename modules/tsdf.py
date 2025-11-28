import sys
sys.path.insert(0, '/home/nileshk/Research2/volumetricPrimitivesPytorch/')
from torch.autograd import Variable
from modules.transformer import rigidTsdf, rigidPointsTransform
from modules.quatUtils import quat_conjugate
from torch.nn import functional as F
import pdb
import torch

def cuboid_tsdf(sample_points, shape):
    ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 3,
    ## output Batch_size x nP x 3
    nP = sample_points.size(1)
    shape_rep = shape.repeat(1, nP, 1)
    tsdf = torch.abs(sample_points) - shape_rep
    tsdfSq = F.relu(tsdf).pow(2).sum(dim=2)
    return tsdfSq  ## Batch_size x nP x 1

def sphere_tsdf(sample_points, radius):
    ## sample_points Batch_size x nP x 3, shape Batch_size x 1 x 1
    dist = torch.norm(sample_points, dim=2, keepdim=True) ## Batch_size x nP x 1
    tsdf = abs(dist) - radius
    tsdfSq = F.relu(tsdf).pow(2)
    return tsdfSq

def cylinder_tsdf(sample_points, radius, height):
    ## sample_points Batch_size x nP x 3, radius Batch_size x 1 x 1, height Batch_size x 1 x 1
    xy = sample_points[:, :, :2]
    z = sample_points[:, :, 2:3]

    d_xy = torch.norm(xy, dim=2, keepdim=True)
    tsdf_r = F.relu(d_xy - radius).pow(2)
    tsdf_h = F.relu(abs(z) - height).pow(2)
    tsdf = tsdf_r + tsdf_h
    return tsdf