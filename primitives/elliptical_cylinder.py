import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class EllipticalCylinderSurface:

    def __init__(self, scale, quaternion, translation):
        self.scale = scale
        self.quaternion = quaternion
        self.translation = translation
        self.is_positive = True

        self.min_xyz, self.max_xyz = self._get_bounds()

    def _get_bounds(self):
        # radii and height
        rx, h, rz = self.scale
        R = quaternion_to_matrix(self.quaternion)   # (3,3)

        # --- 1. Height (local y-axis)
        # world-space contribution of +/- h/2 along local Y
        # height_half[i] = abs(R[i,1]) * (h/2)
        height_half = torch.abs(R[:, 1]) * (h / 2)

        # --- 2. Ellipse in XZ (local)
        # Extreme point along world axis i:
        # ellipse_half[i] = sqrt((rx * R[i,0])^2 + (rz * R[i,2])^2)
        ellipse_half = torch.sqrt(
            (rx * R[:, 0]) ** 2 +
            (rz * R[:, 2]) ** 2
        )

        # Total half extents
        half = ellipse_half + height_half  # (3,)

        # AABB
        min_xyz = self.translation - half
        max_xyz = self.translation + half

        return min_xyz, max_xyz

    def elliptical_cylinder_sdf(self, points):
        """
        SDF for an elliptical cylinder (aligned with local y-axis).
        points: (N, 3) tensor of query points
        center: (3,) tensor
        scale: (3,) tensor [x_radius, height, z_radius]
        rotation_quaternion: (4,) tensor [x, y, z, w] (PyTorch3D format)
        Returns: (N,) tensor of signed distances
        """
        # Convert quaternion to rotation matrix using PyTorch3D
        rotation_matrix = quaternion_to_matrix(self.quaternion)
        
        # Transform points to local coordinate system
        points_local = torch.matmul(points - self.translation, rotation_matrix)
        
        # Elliptical cross-section in XZ plane
        x_radius = self.scale[0]
        height = self.scale[1]
        z_radius = self.scale[2]
        
        # Normalize XZ coordinates by their respective radii to get distance from ellipse
        xz_normalized = torch.stack([
            points_local[..., 0] / x_radius,
            points_local[..., 2] / z_radius
        ], dim=-1)
        
        d_xz = torch.norm(xz_normalized, dim=-1) - 1.0
        # Scale back to world space (approximate)
        d_xz = d_xz * torch.min(torch.tensor([x_radius, z_radius]))
        
        # Distance along Y axis (height)
        d_y = torch.abs(points_local[..., 1]) - height / 2.0
        
        # Combine distances
        d = torch.stack([d_xz, d_y], dim=-1)
        outside_dist = torch.norm(torch.clamp(d, min=0.0), dim=-1)
        inside_dist = torch.clamp(torch.max(d, dim=-1)[0], max=0.0)
        
        return outside_dist + inside_dist

    def __call__(self, points):
        return self.elliptical_cylinder_sdf(points), self.min_xyz, self.max_xyz