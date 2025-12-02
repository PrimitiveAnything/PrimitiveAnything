import torch
from pytorch3d.structures import Volumes
from utils.marching_cubes import marching_cubes_batch
from torch.nn.utils.rnn import pad_sequence

def compute_combined_sdf_from_primitives(grid_points: torch.Tensor, primitives: list[list]):
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
        if positives_distances:
            positives_distances = torch.stack(positives_distances) # (P, N)
            batch_positive_distances.append(positives_distances)
        if negatives_distances:
            negatives_distances = torch.stack(negatives_distances) # (P, N)
            batch_negative_distances.append(negatives_distances)
    if batch_positive_distances:
        batch_positive_distances = torch.stack(batch_positive_distances) # (B, P, N)
    else:
        batch_positive_distances = None
    if batch_negative_distances:
        batch_negative_distances = torch.stack(batch_negative_distances) # (B, P, N)
    else:
        batch_negative_distances = None
    
    if batch_negative_distances is None and batch_positive_distances is None:
        return None
    else:
        combined_sdf = combine_sdfs(batch_positive_distances, batch_negative_distances) # (B, N)

    return combined_sdf

def combine_sdfs(positive_distances: torch.Tensor | None, negative_distances: torch.Tensor | None):
    """
    Combine multiple SDFs using CSG operations.
    positive_distances: (B: Batch, P: nPrimitives, N: nPoints)
    negative_distances: (B: Batch, P: nPrimitives, N: nPoints)
    Returns: (N,) tensor of combined SDF values
    """
    if positive_distances is None and negative_distances is None:
        raise ValueError("Need at least one primitive")
    
    # Union of primitives (min operation)
    if positive_distances is not None:
        result_sdf = positive_distances.min(dim=1)[0] # (B, P, N) -> (B, N)
    elif negative_distances:
        # If no positive primitives, start with large positive values
        B, _, N = negative_distances.shape
        result_sdf = torch.full((B, N), float('inf'))
    
    if negative_distances is not None:
        negative_distances = negative_distances.min(dim=1)[0] # (B, P, N) -> (B, N)
        result_sdf = torch.max(result_sdf, -negative_distances) # (B, N), (B, N) -> (B, N)
    
    return result_sdf

def generate_mesh_from_volumes(volumes: Volumes, device: str = 'cpu'):
    """
    Generate a mesh from primitive volumes using marching cubes.
    
    volumnes (Volumes): 
    
    Returns: vertices, faces
    """
    # Extract mesh using marching cubes
    resolution = volumes.densities().shape[-1]
    vertices, faces = marching_cubes_batch(volumes.densities(), iso=0.5)
    batch_vertices = pad_sequence(vertices, batch_first=True, padding_value=0)
    batch_faces = pad_sequence(faces, batch_first=True, padding_value=-1)
    # Rescale the vertices
    batch_vertices = batch_vertices / resolution * 2 - 1
    batch_vertices = volumes.local_to_world_coords(batch_vertices)
    
    return batch_vertices, batch_faces

def generate_volume_from_primitives(primitives_batch: list[list], device: str = 'cpu', resolution=128):
    """
    Generate a PyTorch3D Volumes object from primitives.
    
    params_tensor: (N, 10) tensor with [scale(3), rotation(4), translation(3)]
    types_tensor: (N,) tensor with primitive type indices
    resolution: int, grid resolution for the volume
    bbox_min, bbox_max: bounding box for the grid
    
    Returns: pytorch3d.structures.Volumes object
    """
    B = len(primitives_batch)
    assert B > 0, "primitives must not be an empty list"
    seq_lengths = [len(primitive_list) for primitive_list in primitives_batch]
    P = seq_lengths[0]
    assert len(set(seq_lengths)) == 1, "All sequences on primitives_batch must have the same length"

    # Base case: all sequences are empty
    if P == 0:
        occupancy_volume = torch.zeros((B, 1, resolution, resolution, resolution), device=device)
        volumes = Volumes(occupancy_volume)
        return volumes

    # Determine the xyz bounds for the batch
    xyz_min = []
    xyz_max = []
    for primitive_list in primitives_batch:
        for primitive in primitive_list:
            xyz_min.append(primitive.min_xyz.cpu())
            xyz_max.append(primitive.max_xyz.cpu()) 
    xyz_min = torch.min(torch.stack(xyz_min), dim=0)[0]
    xyz_max = torch.max(torch.stack(xyz_max), dim=0)[0]

    # Create grid points to evaluate on the combined SDF
    x = torch.linspace(xyz_min[0].item(), xyz_max[0].item(), resolution, device=device)
    y = torch.linspace(xyz_min[1].item(), xyz_max[1].item(), resolution, device=device)
    z = torch.linspace(xyz_min[2].item(), xyz_max[2].item(), resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    # Get the SDF values for the composed primitives
    sdf_values = compute_combined_sdf_from_primitives(
        grid_points=grid_points, primitives=primitives_batch
    ) # (B, N)
    assert sdf_values is not None, "compute_combined_sdf_from_primitives was called with no primitives"

    # Convert SDF to occupancy (negative SDF = inside = 1, positive SDF = outside = 0)
    B, N = sdf_values.shape
    sdf_volume = sdf_values.reshape(B, 1, resolution, resolution, resolution)
    occupancy_volume = (sdf_volume < 0).float() # Occupied if inside the composed shape
    occupancy_volume = occupancy_volume.permute(0, 1, 4, 3, 2) # (B, C, D, H, W)
    
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

def generate_mesh_from_primitives(primitives: list[list], device: str = 'cpu', resolution=128):
    volumes = generate_volume_from_primitives(primitives_batch=primitives, device=device, resolution=resolution)
    meshes = generate_mesh_from_volumes(volumes=volumes, device=device)
    return meshes