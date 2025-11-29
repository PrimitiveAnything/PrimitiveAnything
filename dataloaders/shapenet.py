from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch3d
import pytorch3d.datasets
from pathlib import Path

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)

from pytorch3d.structures import Meshes, Volumes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader

# add path for demo utils functions 
import sys
import os

class ShapeNetDataset(Dataset):
    def __init__(self, shapenet_dir: str = "./data/shapenet/", n_sample_points: int = 1000):
        self.shapenet_dir = shapenet_dir
        self.n_sample_points = n_sample_points

        self.shapenet_dataset = ShapeNetCore(self.shapenet_dir, version=2)

    def __len__(self):
        return len(self.shapenet_dataset)

    def __getitem__(self, index):
        model = self.shapenet_dataset[index]
        mesh = Meshes(verts=[model['verts']], faces=[model['faces']])
        surface_points, normals = sample_points_from_meshes(mesh, num_samples=self.n_sample_points, return_normals=True) # (N, 3)
        surface_points = surface_points.squeeze(0)
        normals = normals.squeeze(0)

        points_normals = torch.cat([surface_points, normals], dim=-1)
        return points_normals # (N, 6), where first three are x, y and z and last three are normals along each axis