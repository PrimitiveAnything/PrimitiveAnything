import argparse
import glob
import json
import yaml
from pathlib import Path
import os
import re
import numpy as np

from scipy.spatial.transform import Rotation
from tqdm import tqdm
import torch
import trimesh

from primitive_anything.primitive_dataset import create_dataset
from primitive_anything.utils import torch_to, count_parameters
from primitive_anything.utils.logger import create_logger, print_log

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

bs_dir = 'data/basic_shapes_norm'
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


def euler_to_quat(euler):
    return Rotation.from_euler('XYZ', euler, degrees=True).as_quat()

def rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(rotvec, degrees=True).as_quat()

def SRT_quat_to_matrix(scale, quat, translation):
    rotation_matrix = Rotation.from_quat(quat).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix * scale
    transform_matrix[:3, 3] = translation
    return transform_matrix

def write_json(primitives, shapename_map, out_path):
    out_json = {}
    out_json['operation'] = 0
    out_json['type'] = 1
    out_json['scene_id'] = None

    new_group = []
    model_scene = trimesh.Scene()
    for scale, rotation, translation, type_code in zip(
        primitives['scale'].squeeze().cpu().numpy(),
        primitives['rotation'].squeeze().cpu().numpy(),
        primitives['translation'].squeeze().cpu().numpy(),
        primitives['type_code'].squeeze().cpu().numpy()
    ):
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

        if new_block['type_id'] == 1101002001034001:
            cur_color = "#2FA9FF"
        elif new_block['type_id'] == 1101002001034002:
            cur_color = "#FFC203"
        elif new_block['type_id'] == 1101002001034010:
            cur_color = "#FF8A9C"

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return np.array([
                int(hex_color[0:2], 16),  # R
                int(hex_color[2:4], 16),  # G
                int(hex_color[4:6], 16),  # B
            ], dtype=np.uint8)[None]

        trans = SRT_quat_to_matrix(scale, euler_to_quat(rotation), translation)
        bs = mesh_bs[bs_name].copy().apply_transform(trans)
        new_vertex_colors = np.repeat(hex_to_rgb(cur_color), bs.visual.vertex_colors.shape[0], axis=0)
        bs.visual.vertex_colors[:, :3] = new_vertex_colors
        vertices = bs.vertices.copy()
        vertices[:, 1] = bs.vertices[:, 2]
        vertices[:, 2] = -bs.vertices[:, 1]
        bs.vertices = vertices
        model_scene.add_geometry(bs)

    out_json['group'] = new_group

    with open(out_path, 'w') as json_file:
        json.dump(out_json, json_file, indent=4)

    glb_path = out_path.replace('.json', '.glb')
    model_scene.export(glb_path)

    return glb_path, out_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/infer.yml', help='Config file path')
    parser.add_argument('-ck', '--AR_ckpt', type=str, default='./ckpt/mesh-transformer.ckpt.60.pt')
    parser.add_argument('-o', '--output', type=str, default='./results/infer')
    parser.add_argument('--bs_dir', type=str, default='data/basic_shapes_norm')
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    bs_names = []
    for bs_path in glob.glob(os.path.join(args.bs_dir, '*.ply')):
        bs_names.append(os.path.basename(bs_path))

    with open(args.config, mode='r') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    AR_checkpoint = torch.load(args.AR_ckpt)
    
    os.makedirs(args.output, exist_ok=True)
    json_result_folder = os.path.join(args.output, 'JsonResults')
    os.makedirs(json_result_folder, exist_ok=True)

    create_logger(Path(args.output))

    dataset = create_dataset(cfg['dataset'])

    transformer = create_model(cfg['model'])
    transformer.load_state_dict(AR_checkpoint)

    for item_i, item in tqdm(enumerate(dataset)):
        pc = item.pop('pc')
    
        item_filename = dataset.data_filename[item_i]
        if torch.cuda.is_available():
            pc = pc.cuda()
            item = torch_to(item, torch.device('cuda'))
            transformer = transformer.cuda()

        recon_primitives, mask = transformer.generate(pc=pc.unsqueeze(0), temperature=args.temperature)

        out_path = os.path.join(json_result_folder, os.path.basename(item_filename).replace('.ply', '.json'))
        write_json(recon_primitives, shapename_map, out_path)
