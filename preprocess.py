import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import faiss
import struct

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *

import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict
import open3d as o3d

color_to_material = {
    (255, 255, 255): "wood",
    (14, 106, 71): "jelly",
    (188, 20, 102): "metal",
    (121, 210, 214): "soil",
    (102, 179, 92): "jelly"
}

color_to_label = {
    (255, 255, 255): "background",
    (14, 106, 71): "plant",
    (188, 20, 102): "pot",
    (121, 210, 214): "water",
    (102, 179, 92): "artifact"
}

material_name_to_index = {
    "sand": 0,
    "soil": 1,
    "metal": 2,
    "jelly": 3,
    "plush": 4,
    "wood": 5,
    "plastic": 6,
    "paste": 7,
    "liquid": 8
}

def show_3d_points_with_colors(points, colors, target_idx=1000):
    """
    points: (N, 3) torch.Tensor
    colors: list of (R, G, B) tuples from vote results
    """
    points = points.detach().cpu().numpy()
    target_point = points[target_idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 正規化顏色為 [0, 1] 的浮點數
    norm_colors = np.array(colors) / 255.0

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=3, c=norm_colors, alpha=0.8)

    # 標記特定點（選用）
    ax.scatter(target_point[0], target_point[1], target_point[2],
               s=50, c='black', label=f"Target Point {target_idx}")

    ax.set_title("Gaussian Point Cloud Colored by Voted Material Color")
    ax.legend()
    plt.show()

def show_3d_points(points):
    points = points.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 所有點用灰色顯示
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='gray', alpha=0.5)

    # 將第 1000 個點標紅
    # ax.scatter(
    #     target_point[0], target_point[1], target_point[2],
    #     s=50, c='red', label=f"Point {target_idx}"
    # )

    ax.set_title("Gaussian Point Cloud with Target Highlighted")
    ax.legend()
    plt.show()

def project_points(particle_x, K, R, t, H = 800):
    # x_cam = R @ x_world + t
    x_cam = (R @ particle_x.T).T + t
    x_img = (K @ x_cam.T).T
    x_img = x_img[:, :2] / x_img[:, 2:3]  # normalize
    x_img[:, 1] = H - x_img[:, 1]
    return x_img

def conver_to_opcv(c2w):
    """Convert camera to OpenCV format."""
    diag = np.diag([1, 1, -1, 1])
    return c2w @ diag

def load_nerf_synthetic_camera_and_images(json_path, image_root):
    image_list = []
    camera_params_list = []

    with open(json_path, 'r') as f:
        meta = json.load(f)

    _frame = meta['frames'][1]
    _image_path = os.path.join(image_root, _frame["file_path"] + ".png")
    _image = cv2.imread(_image_path)
    H, W = _image.shape[:2]

    angle_x = meta["camera_angle_x"]
    f = 0.5 * W / np.tan(0.5 * angle_x)  # focal length
    K = np.array([
        [f, 0, W / 2],
        [0, f, H / 2],
        [0, 0, 1]
    ])

    # for frame in meta["frames"]:
    frame = meta["frames"][0]
    # print(frame)
    # image_path = os.path.join(image_root, frame["file_path"] + ".png")
    image_path = os.path.join("seg_map.png")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_list.append(image)

    transform = np.array(frame["transform_matrix"])
    transform = conver_to_opcv(transform)  # Convert to OpenCV format
    R_c2w = transform[:3, :3]
    t_c2w = transform[:3, 3]
    R = R_c2w.T
    t = -R @ t_c2w
    camera_params_list.append((K, R, t))

    return image_list, camera_params_list

def z_buffer_vote(particle_x, image_list, camera_params_list, max_per_pixel=3, visualize_all=False):
    if isinstance(particle_x, torch.Tensor):
        particle_x = particle_x.detach().cpu().numpy()

    H, W = image_list[0].shape[:2]
    num_particles = particle_x.shape[0]
    color_votes = [dict() for _ in range(num_particles)]
    initialized = np.zeros(num_particles, dtype=bool)

    for img_idx, (img, (K, R, t)) in enumerate(zip(image_list, camera_params_list)):
        proj = project_points(particle_x, K, R, t, H)
        cam_coords = (R @ particle_x.T).T + t
        z_vals = cam_coords[:, 2]  # depth in camera space
        sorted_idx = np.argsort(z_vals)  # near to far

        img_vis = img.copy()

        pixel_map = {}
        for idx in sorted_idx:
            u, v = proj[idx]
            x, y = int(u), int(v)
            if 0 <= x < W and 0 <= y < H:
                px_key = (x, y)
                if px_key not in pixel_map:
                    pixel_map[px_key] = []
                if len(pixel_map[px_key]) < max_per_pixel:
                    pixel_map[px_key].append(idx)
                    color = tuple(img[y, x])
                    color_votes[idx][color] = color_votes[idx].get(color, 0) + 1
                    initialized[idx] = True

                if visualize_all:
                    cv2.circle(img_vis, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
        
        if visualize_all:
            plt.imshow(img_vis)
            plt.title(f"Camera Projected Gaussian Points")
            plt.axis('off')
            plt.show()

    return color_votes, initialized

def knn_infill(particle_x, initialized_mask, color_votes, color_to_material, k=5):
    num_particles = len(particle_x)
    colors = [(255, 255, 255)] * num_particles
    for i, vote in enumerate(color_votes):
        if vote:
            colors[i] = max(vote.items(), key=lambda x: x[1])[0]

    known_idx = np.where(initialized_mask)[0]
    unknown_idx = np.where(~initialized_mask)[0]
    print(f"Total particles: {num_particles}")
    print(f"Known particles: {len(known_idx)}, Unknown particles: {len(unknown_idx)}")

    if isinstance(particle_x, torch.Tensor):
        particle_x = particle_x.detach().cpu().numpy()
    if isinstance(known_idx, torch.Tensor):
        known_idx = known_idx.cpu().numpy()
    tree = cKDTree(particle_x[known_idx])
    _, knn_idx = tree.query(particle_x[unknown_idx], k=k)

    for i, idx in enumerate(unknown_idx):
        neighbors = knn_idx[i]
        neighbor_colors = [colors[known_idx[n]] for n in neighbors]
        # majority vote
        color = max(set(neighbor_colors), key=neighbor_colors.count)
        colors[idx] = color

    # Convert to tensors
    # E_list, nu_list, density_list = [], [], []
    # material_names = []
    # material_count = defaultdict(int)
    # for color in colors:
    #     mat = color_to_material.get(color, "jelly")
    #     material_names.append(mat)
    #     material_count[mat] += 1
        # p = param_table[mat]
        # E_list.append(p["E"])
        # nu_list.append(p["nu"])
        # density_list.append(p["density"])

    material_indices = [material_name_to_index[color_to_material.get(color, "jelly")] for color in colors]
    material_tensor = torch.tensor(material_indices, dtype=torch.long, device="cuda")

    print("Material assignment statistics:")
    # for mat, count in material_count.items():
    #     print(f"{mat}: {count} gaussians")

    return material_tensor, colors

def knn_fill_new_physics_dict(init_pos, final_pos, material_dict, k=3):
    """
    對 material_dict 裡每個欄位做 KNN 補點，並回傳新的 dict
    """
    N = init_pos.shape[0]
    M = final_pos.shape[0] - N

    if M <= 0:
        print("No new Gaussians added.")
        return material_dict  # 原封不動返回

    # KNN 查鄰居
    new_pos = final_pos[N:].detach().cpu().numpy()
    orig_pos = init_pos.detach().cpu().numpy()
    tree = cKDTree(orig_pos)
    _, knn_indices = tree.query(new_pos, k=k)

    filled_dict = {}
    for key, tensor in material_dict.items():
        # shape: (N,)
        data = tensor.detach().cpu()
        if data.dtype == torch.long:
            # 取最近鄰眾數（mode）
            neighbors = data[knn_indices]  # (M, k)
            mode_vals = torch.mode(neighbors, dim=1).values  # (M,)
            new_data = torch.cat([data, mode_vals], dim=0)
        else:
            # 取平均
            avg_vals = data[knn_indices].mean(dim=1)  # (M,)
            new_data = torch.cat([data, avg_vals], dim=0)

        filled_dict[key] = new_data.to(tensor.device)

    return filled_dict

def save_gaussians_with_saveply(
    output_dir, label_to_indices, gaussians, phys_dict
):
    os.makedirs(output_dir, exist_ok=True)

    for label, idxs in label_to_indices.items():
        if not idxs:
            continue
        idxs = np.asarray(idxs)

        g_sub = build_subset_gaussian_from_model(gaussians, idxs)

        ply_path = os.path.join(output_dir, f"{label}.ply")
        g_sub.save_ply(ply_path)           # 若 fork 過，可加 as_binary=True

        # 物理參數 json
        phys_subset = {}
        for k in phys_dict:
            v = phys_dict[k]
            if isinstance(v, torch.Tensor):
                phys_subset[k] = v[idxs].tolist()
            else:
                v_np = np.asarray(v)
                phys_subset[k] = v_np[idxs].tolist()

        with open(os.path.join(output_dir, f"{label}_phys.json"), "w") as f:
            json.dump(phys_subset, f, indent=2)

        print(f"[✓] {label}: {len(idxs)} gaussians saved")

def build_subset_gaussian_from_model(g_src: GaussianModel, indices) -> GaussianModel:
    if torch.is_tensor(indices):
        indices = indices.cpu().numpy()
    else:
        indices = np.asarray(indices)

    d = g_src.max_sh_degree          # e.g. 3
    g_sub = GaussianModel(d)

    # ---- 直接 slice ----
    g_sub._xyz           = g_src._xyz.detach().cpu()[indices].clone()
    g_sub._features_dc   = g_src._features_dc.detach().cpu()[indices].clone()
    g_sub._features_rest = g_src._features_rest.detach().cpu()[indices].clone()
    g_sub._opacity       = g_src._opacity.detach().cpu()[indices].clone()
    g_sub._scaling       = g_src._scaling.detach().cpu()[indices].clone()
    g_sub._rotation      = g_src._rotation.detach().cpu()[indices].clone()
    return g_sub

def safe_numpy(x):
    return None if x is None else x.detach().cpu().numpy()

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--camera_config", type=str, required=True)
    parser.add_argument("--camera_list", type=str, required=True)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)


    project_pos = params["pos"]
    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
    
    print("Load image list and camera parameters list")
    image_list, camera_params_list = load_nerf_synthetic_camera_and_images(args.camera_config, 
                                                                           args.camera_list)
    
    # origin_pos = mpm_init_pos.clone()
    # origin_pos = apply_inverse_rotations(
    #             undotransform2origin(
    #                 undoshift2center111(origin_pos), scale_origin, original_mean_pos
    #             ),
    #             rotation_matrices,
    #         )
    
    print("Project particles to images and vote colors")
    color_votes, initialized = z_buffer_vote(
        particle_x=project_pos,
        image_list=image_list,
        camera_params_list=camera_params_list, 
        max_per_pixel=1
    )

    print("Start infilling material parameters by KNN")
    material_tensor, final_colors = knn_infill(
        particle_x=project_pos,         # same as before
        initialized_mask=initialized,        # bool mask
        color_votes=color_votes,
        color_to_material=color_to_material, 
        k=5
    )

    # show_3d_points(project_pos, target_idx=1000)
    # show_3d_points_with_colors(project_pos, final_colors)

    # 對應每個 Gaussian 的類別名稱
    labels = [color_to_label.get(tuple(c), "plant") for c in final_colors]

    # 分類
    label_to_indices = {k: [] for k in set(labels)}
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    # 組裝資料
    data_dict = {
        "pos": safe_numpy(gaussians.get_xyz),
        "shs": safe_numpy(gaussians.get_features),
        "opacity": safe_numpy(gaussians.get_opacity),
        "scales": safe_numpy(gaussians.get_scaling),
        "rotations": safe_numpy(gaussians.get_rotation)  # 已是 quaternion
    }

    phys_dict = {
        "material": material_tensor
    }

    # Save Gssians with save_ply
    save_gaussians_with_saveply(
        args.output_path, label_to_indices, gaussians, phys_dict
    )

    # gaussians = GaussianModel(3)
    # gaussians.load_ply("output_groups/water.ply")

    # print("xyz:", gaussians.get_xyz.shape)
    # print("opacity:", gaussians.get_opacity.shape)
    # print("sh:", gaussians.get_features.shape)
    # print("scale:", gaussians.get_scaling.shape)
    # print("rotation:", gaussians.get_rotation.shape)
    # print("has NaN in xyz:", torch.isnan(gaussians.get_xyz).any())
    # print("has NaN in opacity:", torch.isnan(gaussians.get_opacity).any())
    # print("has NaN in SH:", torch.isnan(gaussians.get_features).any())
    # print("has NaN in scaling:", torch.isnan(gaussians.get_scaling).any())
    # print("has NaN in rotation:", torch.isnan(gaussians.get_rotation).any())