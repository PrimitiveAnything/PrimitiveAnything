import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class CuboidSurface:

    def __init__(self, scale, quaternion, translation):
        self.scale = scale
        self.quaternion = quaternion
        self.translation = translation

    def sdf_cuboid(self, points):
        """
        SDF for a cuboid (box).
        points: (N, 3) tensor of query points
        center: (3,) tensor
        scale: (3,) tensor [height, width, depth]
        rotation_quaternion: (4,) tensor [x, y, z, w] (PyTorch3D format)
        Returns: (N,) tensor of signed distances
        """
        # Convert quaternion to rotation matrix using PyTorch3D
        # PyTorch3D expects (x, y, z, w) format
        rotation_matrix = quaternion_to_matrix(self.quaternion)
        
        # Transform points to local coordinate system
        points_local = torch.matmul(points - self.translation, rotation_matrix)
        
        # Half extents
        half_scale = self.scale / 2.0
        
        # Distance to box
        q = torch.abs(points_local) - half_scale
        outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside_dist = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)
        
        return outside_dist + inside_dist