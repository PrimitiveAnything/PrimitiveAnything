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

class R2N2ShapeNetDataset(Dataset):
    def __init__(self, partition: Literal['train', 'val', 'test'], r2n2_shapenet_dir: str, synsets: list[str] = ['chair'], n_sample_points: int = 1000):
        self.shapenet_dir = Path(r2n2_shapenet_dir, 'shapenet')
        self.r2n2_dir = Path(r2n2_shapenet_dir, 'r2n2')
        self.r2n2_splits_path = Path(r2n2_shapenet_dir, 'split.json')

        self.n_sample_points = n_sample_points

        self.r2n2_dataset = R2N2(partition, str(self.shapenet_dir), str(self.r2n2_dir), str(self.r2n2_splits_path), return_voxels=True, load_textures=False)

    def __len__(self):
        return len(self.r2n2_dataset)

    def __getitem__(self, index):
        model = self.r2n2_dataset[index]
        mesh = Meshes(verts=[model['verts']], faces=[model['faces']])
        surface_points = sample_points_from_meshes(mesh, num_samples=self.n_sample_points).squeeze(0) # (N, 3)
        # TODO: Figure out which voxel to use (we have 1 per view)
        voxel = model['voxels'][0].unsqueeze(0).float() # (V, D, D, D) where V is the number of views
        return voxel, surface_points
    
    def collate_fn(self, batch) -> tuple[Volumes, torch.Tensor]:
        voxels, points = zip(*batch)
        voxels = torch.stack(voxels) # (1, D, D, D) -> (N, 1, D, D, D)
        voxels = Volumes(densities=voxels)
        points = torch.stack(points) # (S, 3) -> (N, S, 3)
        return voxels, points