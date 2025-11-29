import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class CuboidSurface:

    def __init__(self, scale, quaternion, translation):
        self.scale = scale
        self.quaternion = quaternion
        self.translation = translation
        self.is_positive = True

        self.min_xyz, self.max_xyz = self._get_bounds()

    def _get_bounds(self):
        """
        Returns (min_xyz, max_xyz) of the cuboid in WORLD coordinates.
        Useful for voxelization.
        """
        # 1. Get the 8 corners in local space
        half = self.scale / 2.0
        corners_local = torch.tensor([
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1],  half[2]],
            [-half[0],  half[1], -half[2]],
            [-half[0],  half[1],  half[2]],
            [ half[0], -half[1], -half[2]],
            [ half[0], -half[1],  half[2]],
            [ half[0],  half[1], -half[2]],
            [ half[0],  half[1],  half[2]],
        ], dtype=self.scale.dtype, device=self.scale.device)  # (8,3)

        # 2. Rotate corners to world space
        R = quaternion_to_matrix(self.quaternion)   # (3,3)
        corners_world = corners_local @ R.T          # local → world

        # 3. Translate
        corners_world = corners_world + self.translation   # (8,3)

        # 4. AABB min/max
        min_xyz = corners_world.min(dim=0)[0]
        max_xyz = corners_world.max(dim=0)[0]

        return min_xyz, max_xyz

    def cuboid_sdf(self, points):
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
    
    def __call__(self, points):
        return self.cuboid_sdf(points), self.min_xyz, self.max_xyz