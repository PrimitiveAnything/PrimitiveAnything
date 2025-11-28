import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

class EllipsoidSurface:

    def __init__(self, scale, quaternion, translation):
        self.scale = scale
        self.quaternion = quaternion
        self.translation = translation

    def ellipsoid_sdf(self, points):
        """
        SDF for an ellipsoid.
        points: (N, 3) tensor of query points
        Returns: (N,) tensor of signed distances
        """
        rotation = quaternion_to_matrix(self.quaternion)
        # Transform to unit sphere space by dividing by scale
        points = points - self.translation
        points = (rotation.tranpose(-2, -1) @ points.T).T
        normalized_points = points / self.scale

        # Distance in normalized space
        dist_normalized = torch.norm(normalized_points, dim=-1)
        # Approximate SDF (exact SDF for ellipsoid is more complex)
        # This approximation works well for most cases
        dist = (dist_normalized - 1.0) * torch.min(self.scale)
        return dist