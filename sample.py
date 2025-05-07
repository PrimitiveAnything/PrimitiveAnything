import argparse
from functools import partial
import glob
import multiprocessing
import os
import time

from mesh_to_sdf import get_surface_point_cloud
import numpy as np
import open3d as o3d
import trimesh

os.environ["PYOPENGL_PLATFORM"] = "egl"


def sample_surface_points(mesh, number_of_points=500000, surface_point_method="scan", sign_method="normal",
                          scan_count=100, scan_resolution=400, sample_point_count=10000000, return_gradients=False,
                          return_surface_pc_normals=False, normalized=False):
    sample_start = time.time()
    if surface_point_method == "sample" and sign_method == "depth":
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = "normal"

    surface_start = time.time()
    bound_radius = 1 if normalized else None
    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, bound_radius, scan_count, scan_resolution,
                                                  sample_point_count,
                                                  calculate_normals=sign_method == "normal" or return_gradients)

    surface_end = time.time()
    print("surface point cloud time cost :", surface_end - surface_start)

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
    print("normal time cost :", normal_end - normal_start)
    sample_end = time.time()
    print("sample surface point time cost :", sample_end - sample_start)
    return surface_points


def process_surface_point(mesh, number_of_near_surface_points, return_surface_pc_normals=False):
    mesh = trimesh.load(mesh, force="mesh")
    surface_point = sample_surface_points(mesh, number_of_near_surface_points, return_surface_pc_normals=return_surface_pc_normals)
    return surface_point


def sample_model(model_path, num_points, return_surface_pc_normals=True):
    pc_out_path = os.path.join(args.output_dir, os.path.basename(model_path)).replace(f".{args.postfix}", ".ply")
    if os.path.exists(pc_out_path):
        print(f"{pc_out_path}: exists!")
        return

    try:
        surface_point = process_surface_point(model_path, num_points, return_surface_pc_normals=return_surface_pc_normals)

        coords = surface_point[:, :3]
        normals = surface_point[:, 3:]

        assert (np.linalg.norm(np.asarray(normals), axis=-1) > 0.99).all()
        assert (np.linalg.norm(np.asarray(normals), axis=-1) < 1.01).all()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coords)*0.5)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(pc_out_path, pcd)
        print(f"write_point_cloud: {pc_out_path}")
    except:
        print(f"[ERROR] file: {pc_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./results/infer/JsonResults")
    parser.add_argument("--output_dir", type=str, default="./results/infer/PointClouds")
    parser.add_argument("--num_points", type=int, default=10000)
    parser.add_argument("--postfix", type=str, default="glb")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Invalid input!")
        exit(1)

    if os.path.exists(args.output_dir):
        print(f"path: {args.output_dir} exists!")
        # exit(1)
    else:
        os.makedirs(args.output_dir)

    model_prefix = os.path.join(args.input_dir, f"*.{args.postfix}")
    model_path_list = sorted(list(glob.glob(model_prefix)))

    sample_model_func = partial(sample_model, num_points=args.num_points, return_surface_pc_normals=True)
    with multiprocessing.Pool(16) as pool:
        pool.map(sample_model_func, model_path_list)