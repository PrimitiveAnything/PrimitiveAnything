import torch
from torch.autograd import Variable
import pytorch3d.ops
from pytorch3d.structures import Volumes
import torch.nn.functional as F

from modules.cuboid import CuboidSurface
from modules.transformer import rigidPointsTransform



def consistency_loss(predParts, n_samples, targetPoints, loadedVoxels):
    cuboid_sampler = CuboidSurface(n_samples, normFactor="Surf")
    sampled_points, imp_weights = partComposition(predParts, cuboid_sampler)
    norm_weights = normalize_weights(imp_weights)
    # Find the closes points
    distance_to_target, _, _ = pytorch3d.ops.knn_points(sampled_points, targetPoints, K=1) # (N, SampleSize, K)
    # Check if the predicted points are inside the target volume
    is_inside = is_point_inside_voxel_grid(sampled_points, loadedVoxels)
    # Set distance to zero if inside the target volume
    distance_to_target[is_inside] = 0
    weighted_loss = (distance_to_target * norm_weights).mean()  # B x nP x 1
    return weighted_loss

def is_point_inside_voxel_grid(points_world: torch.Tensor, volumes: Volumes) -> torch.BoolTensor:
    """
    Checks if 3D points are inside the spatial bounds of a PyTorch3D Volumes object.

    Args:
        points_world (torch.Tensor): A tensor of shape (N, 3) representing
                                     world coordinates of points.
        volumes_obj (Volumes): The PyTorch3D Volumes object defining the grid.

    Returns:
        torch.Tensor: A boolean tensor of shape (N,) where True indicates
                      the point is inside the volume's spatial bounds.
    """
    # 1. Transform the points to local coordinate frame
    points_local = volumes.world_to_local_coords(points_world)

    B, N, C = points_local.shape

    # F.grid_sample expects grid coordinates in (x, y, z) order matching (W, H, D)
    # Our voxel is (B, C, depth, height, width)
    # So we need to reorder: (x, y, z) -> (z, y, x) for grid_sample
    grid = points_local.view(B, N, 1, 1, 3)
    grid = grid[..., [2, 1, 0]]  # Flip to match grid_sample's coordinate system
    
    # Sample from voxel
    # Output shape: (B, 1, N, 1, 1)
    sampled = F.grid_sample(
        volumes.densities(), 
        grid, 
        mode='nearest',  # No interpolation
        padding_mode='zeros',  # Values outside voxel are 0
        align_corners=True
    )
    
    # Reshape to (B, N, 1)
    values = sampled.view(B, N, 1) > 0.5
    
    return values

def partComposition(predParts, cuboid_sampler):
    # B x nParts x 10
    nParts = predParts.size(1)
    all_sampled_points = []
    all_sampled_weights = []
    predParts = torch.chunk(predParts, nParts, 1)
    for i in range(nParts):
        sampled_points, imp_weights = primtive_surface_samples(
            predParts[i], cuboid_sampler
        )
        transformedSamples = transform_samples(
            sampled_points, predParts[i]
        )  # B x nPs x 3
        all_sampled_points.append(transformedSamples)  # B x nPs x 3
        all_sampled_weights.append(imp_weights)

    pointsOut = torch.cat(all_sampled_points, dim=1)  # b x nPs*nParts x 3
    weightsOut = torch.cat(all_sampled_weights, dim=1)  # b x nPs*nParts x 1
    return pointsOut, weightsOut


def primtive_surface_samples(predPart, cuboid_sampler):
    # B x 1 x 10
    shape = predPart[:, :, 0:3]  # B  x 1 x 3
    probs = predPart[:, :, 11:12]  # B x 1 x 1
    samples, imp_weights = cuboid_sampler.sample_points_cuboid(shape)
    probs = probs.expand(imp_weights.size())
    imp_weights = imp_weights * probs
    return samples, imp_weights


def transform_samples(samples, predParts):
    # B x nSamples x 3  , predParts B x 1 x 10
    trans = predParts[:, :, 3:6]  # B  x 1 x 3
    quat = predParts[:, :, 6:10]  # B x 1 x 4
    transformedSamples = rigidPointsTransform(samples, trans, quat)
    return transformedSamples


def normalize_weights(imp_weights):
    # B x nP x 1
    totWeights = (torch.sum(imp_weights, dim=1, keepdim=True) + 1e-6)
    norm_weights = imp_weights / totWeights
    return norm_weights


def chamfer_forward(queryPoints, loadedCPs, gridBound, gridSize, loadedVoxels):
    # query points is B x nQ x 3
    neighbourIds = pointClosestCellIndex(queryPoints, gridBound, gridSize).data
    loadedCPs = Variable(loadedCPs.cuda())
    queryDiffs = []

    for b in range(queryPoints.size(0)):
        inds = neighbourIds[b]
        inds = gridSize * gridSize * inds[:, 0] + gridSize * inds[:, 1] + inds[:, 2]
        cp = loadedCPs[b, 0].view(-1, 3)
        cp = cp[inds]
        voxels = Variable(loadedVoxels[b][0].view(-1))
        voxels = voxels[inds]
        diff = (cp - queryPoints[b].view(-1, 3)).pow(2).sum(1)
        queryDiffs.append((-voxels + 1) * diff)
    queryDiffs = torch.stack(queryDiffs)

    return queryDiffs


def pointClosestCellIndex(points, gridBound, gridSize):
    gridMin = -gridBound + gridBound / gridSize
    gridMax = gridBound - gridBound / gridSize
    inds = (points - gridMin) * gridSize / (2 * gridBound)
    inds = torch.round(torch.clamp(inds, min=0, max=gridSize - 1)).long()
    return inds
