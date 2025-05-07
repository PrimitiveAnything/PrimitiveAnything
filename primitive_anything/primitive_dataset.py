import copy
import json
import os

import numpy as np  
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
from torch.utils.data import Dataset

from .utils import exists
from .utils.logger import print_log


def create_dataset(cfg_dataset):
    kwargs = cfg_dataset
    name = kwargs.pop('name')
    dataset = get_dataset(name)(**kwargs)
    print_log(f"Dataset '{name}' init: kwargs={kwargs}, len={len(dataset)}")
    return dataset

def get_dataset(name):
    return {
        'base': PrimitiveDataset,
    }[name]


SHAPE_CODE = {
    'CubeBevel': 0,
    'SphereSharp': 1,
    'CylinderSharp': 2,
}


class PrimitiveDataset(Dataset): 
    def __init__(self,
        pc_dir,
        bs_dir,
        max_length=144,
        range_scale=[0, 1],
        range_rotation=[-180, 180],
        range_translation=[-1, 1],
        rotation_type='euler',
        pc_format='pc',
    ):
        self.data_filename = os.listdir(pc_dir)

        self.pc_dir = pc_dir
        self.max_length = max_length
        self.range_scale = range_scale
        self.range_rotation = range_rotation
        self.range_translation = range_translation
        self.rotation_type = rotation_type
        self.pc_format = pc_format

        with open(os.path.join(bs_dir, 'basic_shapes.json'), 'r', encoding='utf-8') as f:
            basic_shapes = json.load(f)
            
        self.typeid_map = {
            1101002001034001: 'CubeBevel',
            1101002001034010: 'SphereSharp',
            1101002001034002: 'CylinderSharp',
        }

    def __len__(self):
        return len(self.data_filename)

    def __getitem__(self, idx):
        pc_file = os.path.join(self.pc_dir, self.data_filename[idx])
        pc = o3d.io.read_point_cloud(pc_file)

        model_data = {}

        points = torch.from_numpy(np.asarray(pc.points)).float()
        colors = torch.from_numpy(np.asarray(pc.colors)).float()
        normals = torch.from_numpy(np.asarray(pc.normals)).float()
        if self.pc_format == 'pc':
            model_data['pc'] = torch.concatenate([points, colors], dim=-1).T
        elif self.pc_format == 'pn':
            model_data['pc'] = torch.concatenate([points, normals], dim=-1)
        elif self.pc_format == 'pcn':
            model_data['pc'] = torch.concatenate([points, colors, normals], dim=-1)
        else:
            raise ValueError(f'invalid pc_format: {self.pc_format}')

        return model_data
