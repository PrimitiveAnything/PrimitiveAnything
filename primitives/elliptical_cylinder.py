import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class EllipticalCylinderSurface:

    def __init__(self, scale, quaternion, translation):
        self.scale = scale
        self.quaternion = quaternion
        self.translation = translation

    def sdf_elliptical_cylinder(self, points):
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