import os
import time
import glob
import json
import yaml
import torch
import trimesh
import argparse
import mesh2sdf.core
import numpy as np
import skimage.measure
import seaborn as sns
from scipy.spatial.transform import Rotation
from mesh_to_sdf import get_surface_point_cloud
from accelerate.utils import set_seed
from accelerate import Accelerator

from primitive_anything.utils import path_mkdir, count_parameters
from primitive_anything.utils.logger import print_log

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def parse_args():
    parser = argparse.ArgumentParser(description='Process 3D model files')
    
    parser.add_argument(
        '--input',
        type=str,
        default='./data/demo_glb/',
        help='Input file or directory path (default: ./data/demo_glb/)'
    )
    
    parser.add_argument(
        '--log_path',
        type=str,
        default='./results/demo',
        help='Output directory path (default: results/demo)'
    )
    
    return parser.parse_args()

def get_input_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        return glob.glob(os.path.join(input_path, '*'))
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")

args = parse_args()

# Get input files (keeping your original variable name)
input_3ds = get_input_files(args.input)
if not input_3ds:
    raise FileNotFoundError(f"No files found at input path: {args.input}")

# Create output directory (keeping your original variable name)
LOG_PATH = args.log_path
os.makedirs(LOG_PATH, exist_ok=True)

print(f"Found {len(input_3ds)} input files")
print(f"Output directory: {LOG_PATH}")

CODE_SHAPE = {
    0: 'SM_GR_BS_CubeBevel_001.ply',
    1: 'SM_GR_BS_SphereSharp_001.ply',
    2: 'SM_GR_BS_CylinderSharp_001.ply',
}

shapename_map = {
    'SM_GR_BS_CubeBevel_001.ply': 1101002001034001,
    'SM_GR_BS_SphereSharp_001.ply': 1101002001034010,
    'SM_GR_BS_CylinderSharp_001.ply': 1101002001034002,
}

#### config
bs_dir = 'data/basic_shapes_norm'
config_path = './configs/infer.yml'
AR_checkpoint_path = './ckpt/mesh-transformer.ckpt.60.pt'
temperature= 0.0
#### init model
mesh_bs = {}
for bs_path in glob.glob(os.path.join(bs_dir, '*.ply')):
    bs_name = os.path.basename(bs_path)
    bs = trimesh.load(bs_path)
    bs.visual.uv = np.clip(bs.visual.uv, 0, 1)
    bs.visual = bs.visual.to_color()
    mesh_bs[bs_name] = bs

def create_model(cfg_model):
    kwargs = cfg_model
    name = kwargs.pop('name')
    model = get_model(name)(**kwargs)
    print_log("Model '{}' init: nb_params={:,}, kwargs={}".format(name, count_parameters(model), kwargs))
    return model

from primitive_anything.primitive_transformer import PrimitiveTransformerDiscrete
def get_model(name):
    return {
        'discrete': PrimitiveTransformerDiscrete,
    }[name]

with open(config_path, mode='r') as fp:
    AR_train_cfg = yaml.load(fp, Loader=yaml.FullLoader)

AR_checkpoint = torch.load(AR_checkpoint_path)

transformer = create_model(AR_train_cfg['model'])
transformer.load_state_dict(AR_checkpoint)

device = torch.device('cuda')
accelerator = Accelerator(
    mixed_precision='fp16',
)
transformer = accelerator.prepare(transformer)
transformer.eval()
transformer.bs_pc = transformer.bs_pc.cuda()
transformer.rotation_matrix_align_coord = transformer.rotation_matrix_align_coord.cuda()
print('model loaded to device')


def sample_surface_points(mesh, number_of_points=500000, surface_point_method='scan', sign_method='normal',
                          scan_count=100, scan_resolution=400, sample_point_count=10000000, return_gradients=False,
                          return_surface_pc_normals=False, normalized=False):
    sample_start = time.time()
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_start = time.time()
    bound_radius = 1 if normalized else None
    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, bound_radius, scan_count, scan_resolution,
                                                  sample_point_count,
                                                  calculate_normals=sign_method == 'normal' or return_gradients)

    surface_end = time.time()
    print('surface point cloud time cost :', surface_end - surface_start)

    normal_start = time.time()
    if return_surface_pc_normals:
        rng = np.random.default_rng()
        assert surface_point_cloud.points.shape[0] == surface_point_cloud.normals.shape[0]
        indices = rng.choice(surface_point_cloud.points.shape[0], number_of_points, replace=True)
        points = surface_point_cloud.points[indices]
        normals = surface_point_cloud.normals[indices]
        surface_points = np.concatenate([points, normals], axis=-1)
    else:
        surface_points = surface_point_cloud.get_random_surface_points(number_of_points, use_scans=True)
    normal_end = time.time()
    print('normal time cost :', normal_end - normal_start)
    sample_end = time.time()
    print('sample surface point time cost :', sample_end - sample_start)
    return surface_points


def normalize_vertices(vertices, scale=0.9):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale


def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    """
        Convert the non-watertight mesh to watertight.

        Args:
            input_path (str): normalized path
            octree_depth (int):

        Returns:
            mesh(trimesh.Trimesh): watertight mesh

        """
    size = 2 ** octree_depth
    level = 2 / size

    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(normalized_mesh.vertices)
    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)
    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

    # watertight mesh
    vertices = vertices / size * 2 - 1 # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)

    return mesh


def process_mesh_to_surface_pc(mesh_list, marching_cubes=False, dilated_offset=0.0, sample_num=10000):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    for mesh in mesh_list:
        if marching_cubes:
            mesh = export_to_watertight(mesh)
            print("MC over!")
        if dilated_offset > 0:
            new_vertices = mesh.vertices + mesh.vertex_normals * dilated_offset
            mesh.vertices = new_vertices
            print("dilate over!")

        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()

        return_mesh_list.append(mesh)

        pc_normal = np.asarray(sample_surface_points(mesh, sample_num, return_surface_pc_normals=True))
        pc_normal_list.append(pc_normal)
        print("process mesh success")
    return pc_normal_list, return_mesh_list


####    utils
def euler_to_quat(euler):
    return Rotation.from_euler('XYZ', euler, degrees=True).as_quat()

def SRT_quat_to_matrix(scale, quat, translation):
    rotation_matrix = Rotation.from_quat(quat).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix * scale
    transform_matrix[:3, 3] = translation
    return transform_matrix


def write_output(primitives, name):
    out_json = {}
    out_json['operation'] = 0
    out_json['type'] = 1
    out_json['scene_id'] = None

    new_group = []
    model_scene = trimesh.Scene()
    color_map = sns.color_palette("hls", primitives['type_code'].squeeze().shape[0])
    color_map = (np.array(color_map) * 255).astype("uint8")
    for idx, (scale, rotation, translation, type_code) in enumerate(zip(
        primitives['scale'].squeeze().cpu().numpy(),
        primitives['rotation'].squeeze().cpu().numpy(),
        primitives['translation'].squeeze().cpu().numpy(),
        primitives['type_code'].squeeze().cpu().numpy()
    )):
        if type_code == -1:
            break
        bs_name = CODE_SHAPE[type_code]
        new_block = {}
        new_block['type_id'] = shapename_map[bs_name]
        new_block['data'] = {}
        new_block['data']['location'] = translation.tolist()
        new_block['data']['rotation'] = euler_to_quat(rotation).tolist()
        new_block['data']['scale'] = scale.tolist()
        new_block['data']['color'] = ['808080']
        new_group.append(new_block)

        trans = SRT_quat_to_matrix(scale, euler_to_quat(rotation), translation)
        bs = mesh_bs[bs_name].copy().apply_transform(trans)
        new_vertex_colors = np.repeat(color_map[idx:idx+1], bs.visual.vertex_colors.shape[0], axis=0)
        bs.visual.vertex_colors[:, :3] = new_vertex_colors
        vertices = bs.vertices.copy()
        vertices[:, 1] = bs.vertices[:, 2]
        vertices[:, 2] = -bs.vertices[:, 1]
        bs.vertices = vertices
        model_scene.add_geometry(bs)
    out_json['group'] = new_group

    json_path = os.path.join(LOG_PATH, f'output_{name}.json')
    with open(json_path, 'w') as json_file:
        json.dump(out_json, json_file, indent=4)

    glb_path = os.path.join(LOG_PATH, f'output_{name}.glb')
    model_scene.export(glb_path)

    return glb_path, out_json


@torch.no_grad()
def do_inference(input_3d, dilated_offset=0.0, sample_seed=0, do_sampling=False, do_marching_cubes=False, postprocess='none'):
    t1 = time.time()
    set_seed(sample_seed)
    input_mesh = trimesh.load(input_3d, force='mesh')

    # scale mesh
    vertices = input_mesh.vertices
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max() * 1.6
    input_mesh.vertices = vertices

    pc_list, mesh_list = process_mesh_to_surface_pc(
        [input_mesh],
        marching_cubes=do_marching_cubes,
        dilated_offset=dilated_offset
    )
    pc_normal = pc_list[0] # 10000, 6
    mesh = mesh_list[0]

    pc_coor = pc_normal[:, :3]
    normals = pc_normal[:, 3:]

    if dilated_offset > 0:
        # scale mesh and pc
        vertices = mesh.vertices
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max() * 1.6
        mesh.vertices = vertices
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / (bounds[1] - bounds[0]).max() * 1.6

    input_save_name = os.path.join(LOG_PATH, f'processed_{os.path.basename(input_3d)}')
    mesh.export(input_save_name)

    assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), 'normals should be unit vectors, something wrong'
    normalized_pc_normal = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)

    input_pc = torch.tensor(normalized_pc_normal, dtype=torch.float16, device=device)[None]

    with accelerator.autocast():
        if postprocess == 'postprocess1':
            recon_primitives, mask = transformer.generate_w_recon_loss(pc=input_pc, temperature=temperature, single_directional=True)
        else:
            recon_primitives, mask = transformer.generate(pc=input_pc, temperature=temperature)

    output_glb, output_json = write_output(recon_primitives, os.path.basename(input_3d)[:-4])

    return input_save_name, output_glb, output_json


dilated_offset = 0.015
do_marching_cubes = True
postprocess = 'postprocess1'


for input_3d in input_3ds:
    print(f"processing: {input_3d}")
    preprocess_model_obj, output_model_obj, output_model_json = do_inference(
        input_3d,
        dilated_offset=dilated_offset,
        do_marching_cubes=do_marching_cubes,
        postprocess=postprocess
    )
