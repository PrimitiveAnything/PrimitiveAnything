import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.structures import Meshes, Volumes
import pytorch3d.renderer
import json
import argparse
import sys
import mcubes


def sdf_ellipsoid(points, center, scale):
    """
    SDF for an ellipsoid.
    points: (N, 3) tensor of query points
    center: (3,) tensor
    scale: (3,) tensor [x_radius, y_radius, z_radius]
    Returns: (N,) tensor of signed distances
    """
    # Transform to unit sphere space by dividing by scale
    normalized_points = (points - center) / scale
    # Distance in normalized space
    dist_normalized = torch.norm(normalized_points, dim=-1)
    # Approximate SDF (exact SDF for ellipsoid is more complex)
    # This approximation works well for most cases
    dist = (dist_normalized - 1.0) * torch.min(scale)
    return dist


def sdf_cuboid(points, center, scale, rotation_quaternion):
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
    rotation_matrix = quaternion_to_matrix(rotation_quaternion.unsqueeze(0))[0]
    
    # Transform points to local coordinate system
    points_local = torch.matmul(points - center, rotation_matrix)
    
    # Half extents
    half_scale = scale / 2.0
    
    # Distance to box
    q = torch.abs(points_local) - half_scale
    outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside_dist = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)
    
    return outside_dist + inside_dist


def sdf_elliptical_cylinder(points, center, scale, rotation_quaternion):
    """
    SDF for an elliptical cylinder (aligned with local y-axis).
    points: (N, 3) tensor of query points
    center: (3,) tensor
    scale: (3,) tensor [x_radius, height, z_radius]
    rotation_quaternion: (4,) tensor [x, y, z, w] (PyTorch3D format)
    Returns: (N,) tensor of signed distances
    """
    # Convert quaternion to rotation matrix using PyTorch3D
    rotation_matrix = quaternion_to_matrix(rotation_quaternion.unsqueeze(0))[0]
    
    # Transform points to local coordinate system
    points_local = torch.matmul(points - center, rotation_matrix)
    
    # Elliptical cross-section in XZ plane
    x_radius = scale[0]
    height = scale[1]
    z_radius = scale[2]
    
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


def combine_sdfs(sdf_values_list, is_negative_list):
    """
    Combine multiple SDFs using CSG operations.
    sdf_values_list: list of (N,) tensors, each containing SDF values
    is_negative_list: list of booleans indicating if primitive is negative
    Returns: (N,) tensor of combined SDF values
    """
    if len(sdf_values_list) == 0:
        raise ValueError("Need at least one primitive")
    
    # Separate positive and negative primitives
    positive_sdfs = []
    negative_sdfs = []
    
    for sdf_val, is_neg in zip(sdf_values_list, is_negative_list):
        if is_neg:
            negative_sdfs.append(sdf_val)
        else:
            positive_sdfs.append(sdf_val)
    
    # Union of positive primitives (min operation)
    if len(positive_sdfs) > 0:
        combined_positive = torch.stack(positive_sdfs, dim=0)
        result_sdf = torch.min(combined_positive, dim=0)[0]
    else:
        # If no positive primitives, start with large positive values
        result_sdf = torch.full_like(sdf_values_list[0], float('inf'))
    
    # Subtract negative primitives (max with negated SDF)
    for neg_sdf in negative_sdfs:
        result_sdf = torch.max(result_sdf, -neg_sdf)
    
    return result_sdf


def compute_combined_sdf_from_primitives(params_tensor, types_tensor, grid_points):
    """
    Compute combined SDF for primitives represented as tensors.
    
    params_tensor: (N, 10) tensor with [scale(3), rotation(4), translation(3)]
        - scale: [x, y, z] for all primitives
          * cuboid: [width, height, depth]
          * ellipsoid: [x_radius, y_radius, z_radius]
          * elliptical_cylinder: [x_radius, height, z_radius]
    types_tensor: (N,) tensor with primitive type indices
        0: cuboid, 1: ellipsoid, 2: elliptical_cylinder, 
        3: neg_cuboid, 4: neg_ellipsoid, 5: neg_elliptical_cylinder
    grid_points: (M, 3) tensor of query points
    
    Returns: (M,) tensor of combined SDF values
    """
    sdf_values_list = []
    is_negative_list = []
    
    n_primitives = params_tensor.shape[0]
    
    for i in range(n_primitives):
        # Extract parameters
        params = params_tensor[i]
        scale = params[0:3]
        rotation_quat = params[3:7]
        translation = params[7:10]
        
        type_idx = types_tensor[i].item()
        
        # Determine if negative and base type
        is_negative = type_idx >= 3
        base_type_idx = type_idx % 3  # 0: cuboid, 1: ellipsoid, 2: elliptical_cylinder
        
        if base_type_idx == 1:  # ellipsoid
            sdf_val = sdf_ellipsoid(grid_points, translation, scale)
        elif base_type_idx == 0:  # cuboid
            sdf_val = sdf_cuboid(grid_points, translation, scale, rotation_quat)
        elif base_type_idx == 2:  # elliptical_cylinder
            sdf_val = sdf_elliptical_cylinder(grid_points, translation, scale, rotation_quat)
        else:
            raise ValueError(f"Invalid base type index: {base_type_idx}")
        
        sdf_values_list.append(sdf_val)
        is_negative_list.append(is_negative)
    
    return combine_sdfs(sdf_values_list, is_negative_list)


def generate_volume_from_primitives(params_tensor, types_tensor, resolution=128, bbox_min=-3, bbox_max=3):
    """
    Generate a PyTorch3D Volumes object from primitives.
    
    params_tensor: (N, 10) tensor with [scale(3), rotation(4), translation(3)]
    types_tensor: (N,) tensor with primitive type indices
    resolution: int, grid resolution for the volume
    bbox_min, bbox_max: bounding box for the grid
    
    Returns: pytorch3d.structures.Volumes object
    """
    # Create 3D grid
    x = torch.linspace(bbox_min, bbox_max, resolution)
    y = torch.linspace(bbox_min, bbox_max, resolution)
    z = torch.linspace(bbox_min, bbox_max, resolution)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    # Compute combined SDF
    sdf_values = compute_combined_sdf_from_primitives(params_tensor, types_tensor, grid_points)
    sdf_volume = sdf_values.reshape(resolution, resolution, resolution)
    
    # Convert SDF to occupancy (negative SDF = inside = 1, positive SDF = outside = 0)
    # We use a sigmoid to get smooth occupancy values
    occupancy = (sdf_values > 0).float()  # Scale factor controls sharpness
    occupancy_volume = occupancy.reshape(resolution, resolution, resolution)
    
    # Create Volumes object
    # densities should be of shape (batch, density_dim, depth, height, width)
    # We'll use batch size of 1 and density_dim of 1
    densities = occupancy_volume.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W, D)
    
    # Calculate voxel size based on bbox and resolution
    voxel_size = (bbox_max - bbox_min) / (resolution - 1)
    
    # Calculate volume translation (center of the bounding box)
    volume_translation = torch.tensor([
        (bbox_min + bbox_max) / 2,
        (bbox_min + bbox_max) / 2,
        (bbox_min + bbox_max) / 2
    ])
    
    # Create Volumes object
    volumes = Volumes(
        densities=densities,
        voxel_size=voxel_size,
        volume_translation=volume_translation
    )
    
    return volumes


def generate_mesh_from_volumes(volumes: Volumes):
    """
    Generate a mesh from primitives using marching cubes.
    
    params_tensor: (N, 10) tensor with [scale(3), rotation(4), translation(3)]
    types_tensor: (N,) tensor with primitive type indices
    resolution: int, grid resolution for marching cubes
    bbox_min, bbox_max: bounding box for the grid
    
    Returns: vertices (V, 3), faces (F, 3)
    """
    # Extract mesh using marching cubes
    volume_array = volumes.densities().cpu().numpy()
    vertices, faces = mcubes.marching_cubes(volume_array.squeeze(), isovalue=0.5)
    
    return vertices, faces

def render_mesh(vertices, faces, device):
    vertices, faces = [torch.from_numpy(x).to(device=device, dtype=torch.float) for x in (vertices, faces)]

    # Define texture
    textures_rgb = torch.ones_like(vertices, device=device) * torch.tensor([0.6, 0.6, 0.3], device=device)
    textures = pytorch3d.renderer.TexturesVertex([textures_rgb])

    # Assemble Mesh
    mesh = Meshes(
        [vertices], [faces], textures
    )

    # Define cameras
    n_views = 10
    distance = 100
    object_center = vertices.mean(dim=-2)
    elevations = 30 # torch.linspace(0, 2 * torch.pi, n_views, device=device).sin() * 30
    rotation_degrees = torch.linspace(-180, 180, n_views, device=device)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=distance, elev=elevations, azim=rotation_degrees, device=device, at=object_center.unsqueeze(0),
    ) # (N, 3, 3), (N, 3)
    cameras = pytorch3d.renderer.cameras.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device,
    )

    # Define lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, distance]], device=device)

    # Initialize renderer
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=256, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings),
        shader=pytorch3d.renderer.HardPhongShader(device=device, lights=lights),
    )
    views = renderer(mesh.extend(n_views), cameras=cameras, lights=lights)
    views = (views[:, :, :, :3] * 255).to(torch.uint8)
    views = views.cpu().numpy()

    imageio.mimwrite(
        'test_output.gif', [img for img in views], frame_duration=80, loop=0
    )


def load_primitives_from_json(json_path):
    """
    Load primitives from a JSON file.
    
    JSON format should be a list of dictionaries with:
    - 'type': str ('cuboid', 'ellipsoid', 'elliptical_cylinder', 
                   'neg_cuboid', 'neg_ellipsoid', 'neg_elliptical_cylinder')
    - 'scale': list of 3 floats [x, y, z]
        * cuboid: [width, height, depth]
        * ellipsoid: [x_radius, y_radius, z_radius]
        * elliptical_cylinder: [x_radius, height, z_radius] (elliptical base in XZ, height along Y)
    - 'rotation': list of 4 floats [x, y, z, w] quaternion
    - 'translation': list of 3 floats [x, y, z]
    
    Returns: 
        - params_tensor: (N, 10) tensor with [scale(3), rotation(4), translation(3)]
        - types_tensor: (N,) tensor with primitive type indices
            0: cuboid, 1: ellipsoid, 2: elliptical_cylinder, 
            3: neg_cuboid, 4: neg_ellipsoid, 5: neg_elliptical_cylinder
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Mapping from type string to integer
    type_map = {
        'cuboid': 0,
        'ellipsoid': 1,
        'elliptical_cylinder': 2,
        'neg_cuboid': 3,
        'neg_ellipsoid': 4,
        'neg_elliptical_cylinder': 5
    }
    
    params_list = []
    types_list = []
    
    for prim_data in data:
        # Get type index
        prim_type = prim_data['type']
        if prim_type not in type_map:
            raise ValueError(f"Unknown primitive type: {prim_type}. "
                           f"Valid types: {list(type_map.keys())}")
        type_idx = type_map[prim_type]
        
        # Concatenate scale, rotation, translation into single vector
        scale = prim_data['scale']
        rotation = prim_data['rotation']
        translation = prim_data['translation']
        
        # Validate that we have exactly 3 scale values
        if len(scale) != 3:
            raise ValueError(f"Scale must have exactly 3 values [x, y, z], got {len(scale)}")
        if len(rotation) != 4:
            raise ValueError(f"Rotation must have exactly 4 values, got {len(scale)}")
        if len(translation) != 3:
            raise ValueError(f"Translation must have exactly 3 values, got {len(scale)}")
        
        params = scale + rotation + translation  # List concatenation
        params_list.append(params)
        types_list.append(type_idx)
    
    # Convert to tensors
    params_tensor = torch.tensor(params_list, dtype=torch.float32)  # (N, 10)
    types_tensor = torch.tensor(types_list, dtype=torch.long)  # (N,)
    
    return params_tensor, types_tensor


def main():
    """
    Main function to generate and visualize primitives from a JSON file.
    
    Usage: python script.py <path_to_json>
    
    JSON format example:
    [
        {
            "type": "ellipsoid",
            "scale": [0.8, 0.6, 0.8],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0]
        },
        {
            "type": "neg_elliptical_cylinder",
            "scale": [0.3, 1.0, 0.2],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0]
        }
    ]
    
    Scale interpretation:
    - cuboid: [width, height, depth] - half-extents in each direction
    - ellipsoid: [x_radius, y_radius, z_radius] - radii along each axis
    - elliptical_cylinder: [x_radius, height, z_radius] - elliptical base in XZ plane, height along Y
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate mesh from primitive composition')
    parser.add_argument('json_path', type=str, help='Path to JSON file with primitive definitions')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution for voxel and marching cubes')
    parser.add_argument('--output', type=str, default='primitive_composition.png', help='Output image path')
    
    args = parser.parse_args()
    
    # Load primitives from JSON
    try:
        params_tensor, types_tensor = load_primitives_from_json(args.json_path)
        print(f"Loaded {params_tensor.shape[0]} primitives from {args.json_path}")
        print(f"Parameters tensor shape: {params_tensor.shape}")
        print(f"Types tensor shape: {types_tensor.shape}")
    except FileNotFoundError:
        print(f"Error: File '{args.json_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{args.json_path}': {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required key in primitive definition: {e}")
        sys.exit(1)
    
    # Generate mesh
    print("Generating volume from primitives...")
    volumes = generate_volume_from_primitives(params_tensor, types_tensor, resolution=args.resolution)

    # Rendering volume
    vertices, faces = generate_mesh_from_volumes(volumes)
    print(f"Generated mesh with {vertices.shape[0]} vertices and {faces.shape[0]} faces")
    render_mesh(vertices, faces, device='cuda')



if __name__ == "__main__":
    main()