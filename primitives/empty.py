import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class EmptySurface:
    def __init__(self, scale=None, quaternion=None, translation=None, is_positive=True):
        self.is_positive = is_positive
        self.min_xyz = torch.zeros((3,))
        self.max_xyz = torch.zeros((3,))

    def empty_sdf(self, points):
        """
        points: (N,3) tensor
        Returns: (N,) tensor
        """
        return torch.ones(points.shape[0], device=points.device, dtype=points.dtype) * 1.0

    def __call__(self, points):
        return self.empty_sdf(points)