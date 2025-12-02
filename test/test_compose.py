import sys
import json
import torch
from argparse import ArgumentParser

from visualization.render_mesh import render_mesh
from primitives.compose import generate_volume_from_primitives, generate_mesh_from_volumes
from utils.get_primitives import get_primitives


def load_primitives_from_json(json_path):
    """
    Load primitives from a JSON file and convert to list of lists of primitive objects.
    
    JSON format should be a list of dictionaries with:
    - 'type': str ('cuboid', 'ellipsoid', 'elliptical_cylinder')
    - 'scale': list of 3 floats [x, y, z]
        * cuboid: [width, height, depth]
        * ellipsoid: [x_radius, y_radius, z_radius]
        * elliptical_cylinder: [x_radius, height, z_radius] (elliptical base in XZ, height along Y)
    - 'rotation': list of 4 floats [x, y, z, w] quaternion
    - 'translation': list of 3 floats [x, y, z]
    
    Returns: 
        - List[List[Primitive]]: A list of lists containing primitive objects (batch of primitives)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Mapping from type string to integer
    type_map = {
        'cuboid': 1,
        'ellipsoid': 2,
        'elliptical_cylinder': 3,
        'neg_cuboid': 4,
        'neg_ellipsoid': 5,
        'neg_elliptical_cylinder': 6,
    }
    
    sample_list = []
    
    for prim_data in data:
        # Get type index
        prim_type = prim_data['type']
        if prim_type not in type_map:
            raise ValueError(f"Unknown primitive type: {prim_type}. "
                           f"Valid types: {list(type_map.keys())}")
        type_idx = type_map[prim_type]
        
        # Extract scale, rotation, translation
        scale = prim_data['scale']
        rotation = prim_data['rotation']
        translation = prim_data['translation']
        
        # Validate that we have exactly 3 scale values
        if len(scale) != 3:
            raise ValueError(f"Scale must have exactly 3 values [x, y, z], got {len(scale)}")
        if len(rotation) != 4:
            raise ValueError(f"Rotation must have exactly 4 values, got {len(rotation)}")
        if len(translation) != 3:
            raise ValueError(f"Translation must have exactly 3 values, got {len(translation)}")
        
        # Create sample vector: [scale(3), rotation(4), translation(3), type(1)]
        sample = scale + rotation + translation + [type_idx]
        sample_list.append(sample)
    
    # Convert to tensor and reshape to (B, T, 11) format expected by get_primitives
    # B=1 (single batch), T=number of primitives, 11=scale(3)+rotation(4)+translation(3)+type(1)
    samples_tensor = torch.tensor([sample_list], dtype=torch.float32)  # (1, N, 11)
    
    # Use get_primitives to convert to list of lists of primitive objects
    primitives = get_primitives(samples_tensor)  # Returns List[List[Primitive]]
    
    return primitives

def test_compose():
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
    
    # Load primitives from JSON
    json_path = 'test/test_compose_input.json'
    output_filename_format = 'test/test_compose_output_{:d}.gif'.format
    resolution = 64
    try:
        primitives = load_primitives_from_json('test/test_compose_input.json')
        print(f"Loaded {len(primitives[0])} primitives from {json_path}")
    except FileNotFoundError:
        print(f"Error: File '{json_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_path}': {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required key in primitive definition: {e}")
        sys.exit(1)
    
    # Generate mesh
    print("Generating volume from primitives...")
    volumes = generate_volume_from_primitives(primitives, resolution=resolution)

    # Rendering volume
    vertices, faces = generate_mesh_from_volumes(volumes)
    print(f"Generated mesh with {vertices.shape[0]} vertices and {faces.shape[0]} faces")
    for batch in range(len(vertices)):
        render_mesh(vertices[batch], faces[batch], output_filename_format(batch), device=volumes.device)