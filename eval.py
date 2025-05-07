import argparse
from collections import defaultdict
import glob
import json
import os

import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm


import numpy as np

def voxelize(points, voxel_size=0.1):
    """
    Converts a set of 3D points to a voxel grid representation.
    Points are quantized to the nearest voxel center.

    Parameters:
    - points: (N, 3) numpy array of 3D points
    - voxel_size: the size of each voxel (adjust this depending on your data)

    Returns:
    - voxels: Set of unique voxel coordinates
    """
    # Quantize the points to the nearest voxel
    quantized_points = np.floor(points / voxel_size).astype(int)
    # Use a set to get unique voxel coordinates
    voxels = set(map(tuple, quantized_points))
    return voxels

def calculate_iou(model_vox, target_vox, voxel_size=0.1):
    """
    Calculate the IoU (Intersection over Union) between two point clouds.

    Parameters:
    - model_vox: (N, 3) numpy array of the first point cloud
    - target_vox: (M, 3) numpy array of the second point cloud
    - voxel_size: Size of the voxels (default is 0.1)

    Returns:
    - iou: Intersection over Union (IoU) score
    """
    # Voxelize both point clouds
    model_voxels = voxelize(model_vox, voxel_size)
    target_voxels = voxelize(target_vox, voxel_size)

    # Calculate intersection and union
    intersection = len(model_voxels.intersection(target_voxels))
    union = len(model_voxels.union(target_voxels))

    # Compute IoU
    iou = intersection / union if union > 0 else 0.0
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./results/infer/PointClouds")
    parser.add_argument("--target_dir", type=str, default="./data/test_pc")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir) or not os.path.exists(args.target_dir):
        print("Invalid input!")
        exit(1)

    model_prefix = os.path.join(args.input_dir, "*.ply")
    model_path_list = sorted(list(glob.glob(model_prefix)))

    distance_json = {}
    distance_list = []
    emu_distance_list = []
    hausdorff_distance_list = []
    voxel_iou_list = []
    for model_path in tqdm(model_path_list):
        model_name = os.path.basename(model_path)
        target_path = os.path.join(args.target_dir, model_name)
        if not os.path.exists(target_path):
            print(f"{target_path}: not found!")
            exit(1)

        model_pc = np.array(trimesh.load(model_path).vertices)
        target_pc = np.array(trimesh.load(target_path).vertices)

        distance = pcu.chamfer_distance(model_pc, target_pc)
        model_pc_downsampled = model_pc[np.random.choice(model_pc.shape[0], 1000, replace=False)]
        target_pc_downsampled = target_pc[np.random.choice(target_pc.shape[0], 1000, replace=False)]
        emu_distance, _ = pcu.earth_movers_distance(model_pc_downsampled, target_pc_downsampled)
        hausdorff_distance = pcu.hausdorff_distance(model_pc, target_pc)

        iou = calculate_iou(model_pc, target_pc, voxel_size=1/32.)

        distance_list.append(distance)
        emu_distance_list.append(emu_distance)
        hausdorff_distance_list.append(hausdorff_distance)
        voxel_iou_list.append(iou)
        model_id = os.path.splitext(model_name)[0]
        distance_json[model_id] = distance

        print(f"{model_id}: chamfer distance: {distance:.3f}, earth movers distance: {emu_distance:.3f}, hausdorff distance: {hausdorff_distance:.3f}, voxel IoU: {iou:.3f}")

    distance_json["mean"] = np.mean(distance_list)
    distance_json["mean_emu"] = np.mean(emu_distance_list)
    distance_json["mean_hausdorff"] = np.mean(hausdorff_distance_list)
    distance_json["mean_voxel_iou"] = np.mean(voxel_iou_list)
    print(f"mean chamfer distance: {np.mean(distance_list)}")
    print(f"mean earth movers distance: {np.mean(emu_distance_list)}")
    print(f"mean hausdorff distance: {np.mean(hausdorff_distance_list)}")
    print(f"mean voxel IoU: {np.mean(voxel_iou_list)}")
    with open(os.path.join(args.input_dir, "distance.json"), "w") as json_file:
        json.dump(distance_json, json_file, indent=4)
