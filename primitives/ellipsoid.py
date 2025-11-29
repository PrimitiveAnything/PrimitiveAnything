import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class EllipsoidSurface:

    def __init__(self, scale, quaternion, translation, is_positive=True):
        self.scale = scale
        self.quaternion = quaternion
        self.translation = translation
        self.is_positive = is_positive

        self.min_xyz, self.max_xyz = self._get_bounds()

    def _get_bounds(self):
        # Get rotation matrix
        R = quaternion_to_matrix(self.quaternion)  # (3,3)

        # Absolute rotation
        absR = torch.abs(R)

        # Half-size of AABB (3,)
        half_extents = absR @ self.scale

        # AABB = center ± half_extents
        min_xyz = self.translation - half_extents
        max_xyz = self.translation + half_extents

        return min_xyz, max_xyz

    def ellipsoid_sdf(self, points):
        """
        SDF for an ellipsoid.
        points: (N, 3) tensor of query points
        Returns: (N,) tensor of signed distances
        """
        rotation = quaternion_to_matrix(self.quaternion)
        # Transform to unit sphere space by dividing by scale
        points = points - self.translation
        points = (rotation.transpose(-2, -1) @ points.T).T
        normalized_points = points / self.scale

        # Distance in normalized space
        dist_normalized = torch.norm(normalized_points, dim=-1)
        # Approximate SDF (exact SDF for ellipsoid is more complex)
        # This approximation works well for most cases
        dist = (dist_normalized - 1.0) * torch.min(self.scale)
        return dist
    
    def __call__(self, points):
        return self.ellipsoid_sdf(points), self.min_xyz, self.max_xyz