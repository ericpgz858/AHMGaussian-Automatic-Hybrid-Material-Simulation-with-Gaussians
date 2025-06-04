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

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)

## my part

color_to_material = {
    (0, 0, 0): "metal",
    (255, 0, 0): "jelly",
    (255, 255, 255): "foam",
    (0, 255, 0): "plasticine",
    (0, 0, 255): "sand",
    (128, 128, 128): "snow"
}

material_param_table = {
    "metal":       {"E": 2e6,  "nu": 0.4, "density": 70},
    "jelly":       {"E": 2e6,  "nu": 0.4, "density": 70},
    "foam":        {"E": 800,  "nu": 0.25, "density": 100},
    "plasticine":  {"E": 3e3,  "nu": 0.45, "density": 1600},
    "sand":        {"E": 5e3,  "nu": 0.35, "density": 1800},
    "snow":        {"E": 1e4,  "nu": 0.3, "density": 900},
}

def show_3d_points(points, target_idx = 1000):
    points = points.detach().cpu().numpy()
    target_point = points[target_idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 所有點用灰色顯示
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='gray', alpha=0.5)

    # 將第 1000 個點標紅
    ax.scatter(
        target_point[0], target_point[1], target_point[2],
        s=50, c='red', label=f"Point {target_idx}"
    )

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

    for frame in meta["frames"]:
        image_path = os.path.join(image_root, frame["file_path"] + ".png")
        if not os.path.exists(image_path):
            continue
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

# def material_params_from_colors(colors, color_to_material, param_table):
#     E_list, nu_list, density_list = [], [], []

#     # See if the first color is in the mapping
#     # first_color = colors[0]
#     # first_material = color_to_material.get(first_color, "jelly")
#     # print(f"[Gaussian 0] Color {first_color} → Material: {first_material}")

#     for color in colors:
#         material = color_to_material.get(color, "jelly")
#         # print(f"Color {color} mapped to material {material}")
#         params = param_table[material]
#         E_list.append(params["E"])
#         nu_list.append(params["nu"])
#         density_list.append(params["density"])
#     return (
#         torch.tensor(E_list, dtype=torch.float32, device=device),
#         torch.tensor(nu_list, dtype=torch.float32, device=device),
#         torch.tensor(density_list, dtype=torch.float32, device=device),
#     )

# def vote_color_from_images(particle_x, image_list, camera_params_list, visualize_all=False):
#     votes = [{} for _ in range(len(particle_x))]

#     for img, (K, R, t) in zip(image_list, camera_params_list):
#         proj = project_points(particle_x, K, R, t)
#         H, W = img.shape[:2]

#         # index = 1000
#         # u0, v0 = proj[index]
#         # x0, y0 = int(u0), int(v0)
#         # print(f"Gaussian[{index}] projected to image[0] pixel: ({x0}, {y0})")

#         # # 圈出那個點
#         img_vis = img.copy()
#         # if 0 <= x0 < W and 0 <= y0 < H:
#         #     cv2.circle(img_vis, (x0, y0), radius=5, color=(255, 0, 0), thickness=2)

#         # plt.imshow(img_vis)
#         # plt.title(f"Image[0] with Gaussian[0] projection {img[y0, x0]} at ({x0}, {y0})")
#         # plt.axis('off')
#         # plt.show()

#         for i, (u, v) in enumerate(proj):
#             x, y = int(u), int(v)
#             if 0 <= x < W and 0 <= y < H:
#                 color = tuple(img[y, x])
#                 votes[i][color] = votes[i].get(color, 0) + 1
              
#             if visualize_all:
#                     cv2.circle(img_vis, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
        
#         if visualize_all:
#             plt.imshow(img_vis)
#             plt.title(f"Camera Projected Gaussian Points")
#             plt.axis('off')
#             plt.show()
        
        
#     # show the votes for the first particle
#     # print("votes at 0:", votes[0])
#     # colors = [max(vote.items(), key=lambda x: x[1])[0] if vote else (255, 255, 255) for vote in votes]
#     # print("colors at 0:", colors[0])

#     return [max(vote.items(), key=lambda x: x[1])[0] if vote else (255, 255, 255) for vote in votes]

# def infer_fill_material_by_knn(
#     mpm_init_pos, gs_num, image_list, camera_params_list,
#     color_to_material, material_param_table, K=5
# ):
#     device = mpm_init_pos.device

#     # Step 1: 投票前 gs_num 的 Gaussian
#     visible_pos = mpm_init_pos[:gs_num]
#     visible_np = visible_pos.detach().cpu().numpy()
#     colors = vote_color_from_images(visible_np, image_list, camera_params_list)
#     known_materials = [color_to_material.get(tuple(c), "jelly") for c in colors]

#     # Step 2: 映射 visible 粒子的材質參數
#     E_main, nu_main, density_main = material_params_from_colors(
#         colors, color_to_material, material_param_table
#     )

#     # Step 3: 使用 torch.cdist 找填充粒子最近鄰
#     fill_pos = mpm_init_pos[gs_num:]  # (M, 3)
#     dists = torch.cdist(fill_pos, visible_pos, p=2)  # (M, gs_num)
#     knn_inds = torch.topk(dists, k=K, largest=False).indices  # (M, K)

#     # Step 4: 多數決決定填充材質
#     E_list, nu_list, density_list = [], [], []
#     for i in range(knn_inds.shape[0]):
#         nbrs = knn_inds[i].cpu().tolist()
#         materials = [known_materials[j] for j in nbrs]
#         mat = max(set(materials), key=materials.count)
#         param = material_param_table[mat]
#         E_list.append(param["E"])
#         nu_list.append(param["nu"])
#         density_list.append(param["density"])

#     E_fill = torch.tensor(E_list, dtype=torch.float32, device=device)
#     nu_fill = torch.tensor(nu_list, dtype=torch.float32, device=device)
#     density_fill = torch.tensor(density_list, dtype=torch.float32, device=device)

#     # Step 5: 合併前後
#     E_tensor = torch.cat([E_main, E_fill], dim=0)
#     nu_tensor = torch.cat([nu_main, nu_fill], dim=0)
#     density_tensor = torch.cat([density_main, density_fill], dim=0)

#     return E_tensor, nu_tensor, density_tensor

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

def knn_infill(particle_x, initialized_mask, color_votes, color_to_material, param_table, k=5):
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
    E_list, nu_list, density_list = [], [], []
    material_count = defaultdict(int)
    for color in colors:
        mat = color_to_material.get(color, "jelly")
        # print(f"Color {color} mapped to material {mat}")
        # if mat == "metal":
        #     print(f"Color {color} mapped to material {mat}")
        material_count[mat] += 1
        p = param_table[mat]
        E_list.append(p["E"])
        nu_list.append(p["nu"])
        density_list.append(p["density"])

    print("Material assignment statistics:")
    for mat, count in material_count.items():
        print(f"{mat}: {count} gaussians")
    device = "cuda:0"
    return (
        torch.tensor(E_list, dtype=torch.float32, device=device),
        torch.tensor(nu_list, dtype=torch.float32, device=device),
        torch.tensor(density_list, dtype=torch.float32, device=device),
    )

## end of my part

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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    visible_gs_num = gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )

    ## My part 將 mpm_init_pos project 到原始的 image 上面
    # show_3d_points(init_pos, target_idx=1000)
    print("Load image list and camera parameters list")
    image_list, camera_params_list = load_nerf_synthetic_camera_and_images("../nerf_synthetic/ficus/transforms_train.json", 
                                                                           "../nerf_synthetic/ficus/")
    
    origin_pos = mpm_init_pos.clone()
    origin_pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(origin_pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
    
    print("Project particles to images and vote colors")
    color_votes, initialized = z_buffer_vote(
        particle_x=origin_pos,
        image_list=image_list,
        camera_params_list=camera_params_list, 
        max_per_pixel=1
    )

    print("Start infilling material parameters by KNN")
    E_tensor, nu_tensor, density_tensor = knn_infill(
        particle_x=origin_pos,         # same as before
        initialized_mask=initialized,        # bool mask
        color_votes=color_votes,
        color_to_material=color_to_material, 
        param_table=material_param_table,
        k=5
    )
        
    # print("Infer material parameters by KNN")
    # E_tensor, nu_tensor, density_tensor = infer_fill_material_by_knn(
    #     init_pos[:visible_gs_num].to(device), visible_gs_num, image_list, camera_params_list,
    #     color_to_material, material_param_table
    # )
    # print("Visible gs number:", visible_gs_num)
    # print("E_tensor:", E_tensor[0])
    # print("nu_tensor:", nu_tensor[0])
    # print("density_tensor:", density_tensor[0])

    print("Set material parameters")
    material_params["per_particle_material"] = {
        "E": E_tensor,
        "nu": nu_tensor,
        "density": density_tensor,
    }
    
    mpm_solver.set_parameters_dict(material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )
