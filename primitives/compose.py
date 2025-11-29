import torch
from pytorch3d.structures import Volumes
import mcubes

def compute_combined_sdf_from_primitives(grid_points: torch.Tensor, primitives: list[list[Primitive]]):
    """
    Compute combined SDF for primitives represented as tensors.
    
    grid_points: (M, 3) tensor of query points
    
    Returns: (M,) tensor of combined SDF values
    """
    batch_positive_distances = []
    batch_negative_distances = []
    for batch in primitives:
        positives_distances = []
        negatives_distances = []
        for primitive in batch:
            distances = primitive(grid_points)
            if primitive.is_positive:           
                positives_distances.append(distances)
            else:
                negatives_distances.append(distances)
        positives_distances = torch.concat(positives_distances) # (P, N)
        negatives_distances = torch.concat(negatives_distances) # (P, N)
        batch_positive_distances.append(positives_distances)
        batch_negative_distances.append(negatives_distances)
    batch_positives_distances = torch.concat(batch_positive_distances) # (B, P, N)
    batch_negatives_distances = torch.concat(batch_negative_distances) # (B, P, N)
    
    return combine_sdfs(batch_positives_distances, batch_negatives_distances) # (B, N)

def combine_sdfs(positive_distances: torch.Tensor, negative_distances: torch.Tensor):
    """
    Combine multiple SDFs using CSG operations.
    positive_distances: (B: Batch, P: nPrimitives, N: nPoints)
    negative_distances: (B: Batch, P: nPrimitives, N: nPoints)
    Returns: (N,) tensor of combined SDF values
    """
    if len(positive_distances) == 0 and len(negative_distances):
        raise ValueError("Need at least one primitive")
    
    # Union of primitives (min operation)
    if len(positive_distances) > 0:
        result_sdf = positive_distances.min(dim=1)[0]
    else:
        # If no positive primitives, start with large positive values
        B, P, N = positive_distances.shape
        result_sdf = torch.full((B, P), float('inf'))
    
    if len(negative_distances):
        negative_distances = negative_distances.min(dim=1)[0] # (B, P, N) -> (B, N)
        result_sdf = torch.max(result_sdf, -negative_distances) # (B, N), (B, N) -> (B, N)
    
    return result_sdf

def generate_mesh_from_volumes(volumes: Volumes):
    """
    Generate a mesh from primitive volumes using marching cubes.
    
    volumnes (Volumes): 
    
    Returns: vertices, faces
    """
    # Extract mesh using marching cubes
    volume_array = volumes.densities().cpu().numpy()
    vertices, faces = mcubes.marching_cubes(volume_array.squeeze(), isovalue=0.5)

    # Rescale the vertices
    vertices = volumes.local_to_world_coords(vertices)
    
    return vertices, faces

def generate_volume_from_primitives(primitives: list[list[Primitive]], resolution=128):
    """
    Generate a PyTorch3D Volumes object from primitives.
    
    params_tensor: (N, 10) tensor with [scale(3), rotation(4), translation(3)]
    types_tensor: (N,) tensor with primitive type indices
    resolution: int, grid resolution for the volume
    bbox_min, bbox_max: bounding box for the grid
    
    Returns: pytorch3d.structures.Volumes object
    """
    # Create 3D grid
    xyz_min = []
    xyz_max = []
    for primitive_list in primitives:
        for primitive in primitive_list:
            xyz_min.append(primitive.min_xyz)
            xyz_max.append(primitive.max_xyz) 
    xyz_min = torch.min(*xyz_min)
    xyz_max = torch.max(*xyz_max)
    x = torch.linspace(xyz_min[0].item(), xyz_max[0].item(), resolution)
    y = torch.linspace(xyz_min[1].item(), xyz_max[1].item(), resolution)
    z = torch.linspace(xyz_min[2].item(), xyz_max[2].item(), resolution)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    # Compute combined SDF
    sdf_values = compute_combined_sdf_from_primitives(
        grid_points=grid_points, primitives=primitives
    ) # (B, N)
    B, N = sdf_values.shape
    sdf_volume = sdf_values.reshape(B, 1, resolution, resolution, resolution)
    
    # Convert SDF to occupancy (negative SDF = inside = 1, positive SDF = outside = 0)
    occupancy_volume = (sdf_volume > 0).float()  # Scale factor controls sharpness
    
    # Create Volumes object
    # densities should be of shape (batch, density_dim, depth, height, width)
    # We'll use batch size of 1 and density_dim of 1

    # Calculate voxel size based on bbox and resolution
    voxel_size = (xyz_max - xyz_min) / (resolution - 1)
    
    # Calculate volume translation (center of the bounding box)
    volume_translation = (xyz_max + xyz_min) / 2
    
    # Create Volumes object
    volumes = Volumes(
        densities=occupancy_volume,
        voxel_size=voxel_size,
        volume_translation=volume_translation
    )
    
    return volumes



