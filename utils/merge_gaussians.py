import sys
from glob import glob
sys.path.append("gaussian-splatting")

import numpy as np
import os
import json
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from utils.render_utils import *
from utils.transformation_utils import *
from utils.decode_param import *
from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings
from utils.render_utils import *

def load_gaussians_and_phys_from_folder(folder_path, sh_degree=3):
    ply_files = [f for f in os.listdir(folder_path) if f.endswith(".ply")]
    all_pos, all_shs, all_opacities = [], [], []
    all_scales, all_rotations = [], []
    all_E, all_nu, all_density = [], [], []

    for ply_file in ply_files:
        label = ply_file[:-4]
        json_path = os.path.join(folder_path, f"{label}_phys.json")
        if not os.path.exists(json_path):
            print(f"[Warning] Missing JSON for {ply_file}, skipping.")
            continue

        g = GaussianModel(sh_degree)
        g.load_ply(os.path.join(folder_path, ply_file))

        all_pos.append(      g.get_xyz.detach().clone())
        all_shs.append(      g.get_features.detach().clone())
        all_opacities.append(g.get_opacity.detach().clone())
        all_scales.append(   g.get_scaling.detach().clone())
        all_rotations.append(g.get_rotation.detach().clone())

        with open(json_path) as f:
            phys = json.load(f)
        all_E.append( torch.tensor(phys["E"], dtype=torch.float32, device="cuda") )
        all_nu.append(torch.tensor(phys["nu"], dtype=torch.float32, device="cuda") )
        all_density.append(torch.tensor(phys["density"], dtype=torch.float32, device="cuda") )

    return {
        "pos":       torch.cat(all_pos,       dim=0),
        "shs":       torch.cat(all_shs,       dim=0),
        "opacity":   torch.cat(all_opacities, dim=0),
        "scales":    torch.cat(all_scales,    dim=0),
        "rotations": torch.cat(all_rotations, dim=0),
        "E":         torch.cat(all_E,         dim=0),
        "nu":        torch.cat(all_nu,        dim=0),
        "density":   torch.cat(all_density,   dim=0),
    }


def save_combined_gaussians_to_ply_json_from_models(
    models: list[GaussianModel], physics_json_paths: list[str], filename_prefix="combined"
):
    assert len(models) == len(physics_json_paths), "Mismatch between models and JSONs"

    all_xyz, all_dc, all_rest = [], [], []
    all_opacity, all_scale, all_rot = [], [], []
    all_E, all_nu, all_density = [], [], []
    all_material = []


    for g, json_path in zip(models, physics_json_paths):
        all_xyz.append(g.get_xyz.detach().cpu())
        # shs = g.get_features.detach().cpu()          # (N, 16, 3)
        # shs = shs.permute(0, 2, 1).contiguous()      # → (N, 3, 16)
        # f_dc = shs[:, :, 0:1]                         # (N, 3, 1)
        # f_rest = shs[:, :, 1:] 
        all_dc.append(g._features_dc.detach().cpu())
        all_rest.append(g._features_rest.detach().cpu())

        all_opacity.append(g._opacity.detach().cpu())       # ✔ 原始值
        all_scale.append(g._scaling.detach().cpu())         # ✔ log-scale
        all_rot.append(g._rotation.detach().cpu())   

        with open(json_path) as f:
            phys = json.load(f)
        all_material.append(torch.tensor(phys["material"], dtype=torch.long))

    # 合併所有資料
    g_combined = GaussianModel(models[0].max_sh_degree)
    g_combined.active_sh_degree = g_combined.max_sh_degree
    g_combined._xyz           = torch.cat(all_xyz, dim=0).clone().to("cuda")
    g_combined._features_dc   = torch.cat(all_dc, dim=0).clone().to("cuda")
    g_combined._features_rest = torch.cat(all_rest, dim=0).clone().to("cuda")
    g_combined._opacity       = torch.cat(all_opacity, dim=0).clone().to("cuda")
    g_combined._scaling       = torch.cat(all_scale, dim=0).clone().to("cuda")
    g_combined._rotation      = torch.cat(all_rot, dim=0).clone().to("cuda")
    # render_gaussian_model(g_combined, frame_num=60)

    # show_3d_points(g_combined.get_xyz)


    # 儲存 .ply
    ply_path = f"{filename_prefix}.ply"
    ply_dir = os.path.dirname(ply_path)
    if ply_dir and not os.path.exists(ply_dir):
        os.makedirs(ply_dir, exist_ok=True)
    g_combined.save_ply(ply_path)
    print(f"[Saved] Gaussian PLY → {ply_path}")

    with open(f"{filename_prefix}_phys.json", "w") as f:
        json.dump({
        "material": torch.cat(all_material, dim=0).tolist()
        }, f, indent=2)
    print(f"[Saved] Physics parameters JSON → {filename_prefix}_phys.json")


def show_3d_points(points):
    points = points.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 所有點用灰色顯示
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='gray', alpha=0.5)
    ax.set_title("Gaussian Point Cloud with Target Highlighted")
    ax.legend()
    plt.show()

def check_nan_inf(name, tensor):
    if torch.isnan(tensor).any():
        print(f"[❌] {name} has NaN")
    if torch.isinf(tensor).any():
        print(f"[❌] {name} has Inf")

def check_gaussian_model_safety(g, visualize_opacity=False):
    print("=== Gaussian Model Safety Check ===")
    print(f"Total Gaussians: {g.get_xyz.shape[0]}")
    
    # Shape Check
    print("xyz      :", g.get_xyz.shape)
    print("features :", g.get_features.shape)
    print("opacity  :", g.get_opacity.shape)
    print("scaling  :", g.get_scaling.shape)
    print("rotation :", g.get_rotation.shape)

    # SH consistency
    expected_SH = (g.max_sh_degree + 1) ** 2
    actual_SH = g.get_features.shape[1]
    if actual_SH != expected_SH:
        print(f"[❌] SH shape mismatch: expected {expected_SH}, got {actual_SH}")
    else:
        print(f"[✓] SH shape correct: {actual_SH}")

    # NaN / Inf check
    check_nan_inf("xyz", g.get_xyz)
    check_nan_inf("opacity", g.get_opacity)
    check_nan_inf("features", g.get_features)
    check_nan_inf("scaling", g.get_scaling)
    check_nan_inf("rotation", g.get_rotation)

    # SH0 stats
    sh0 = g.get_features[:, 0, :]
    print("SH0 mean:", sh0.mean(dim=0).tolist())
    print("SH0 std :", sh0.std(dim=0).tolist())

    # Opacity stats
    opacity = g.get_opacity
    print("Opacity mean:", opacity.mean().item())
    print("Opacity range:", (opacity.min().item(), opacity.max().item()))
    if visualize_opacity:
        plt.hist(opacity.detach().cpu().numpy().flatten(), bins=50)
        plt.title("Opacity Distribution")
        plt.show()

    # Scaling stats
    scale = g.get_scaling
    print("Scale mean:", scale.mean(dim=0).tolist())
    print("Scale std :", scale.std(dim=0).tolist())

    # Covariance stats
    try:
        cov = g.get_covariance().detach().cpu()
        print("Covariance mean abs:", cov.abs().mean().item())
    except Exception as e:
        print(f"[⚠] Failed to compute covariance: {e}")

    # Empty check
    if g.get_xyz.shape[0] == 0:
        print("[❌] Gaussian model is empty!")

    print("=== End of Check ===")

def compare_tensors(name, t1, t2, atol=1e-6):
    same_shape = t1.shape == t2.shape
    same_value = torch.allclose(t1, t2, atol=atol)
    if not same_shape:
        print(f"[❌] {name}: shape mismatch {t1.shape} vs {t2.shape}")
    elif not same_value:
        diff = (t1 - t2).abs()
        print(f"[⚠️] {name}: value mismatch (max diff = {diff.max().item():.6f})")
    else:
        print(f"[✅] {name}: matched")
    
    
ply_paths = sorted(glob("output_groups/*.ply"))
models = []
for path in ply_paths:
    g = GaussianModel(3)  # 確定 degree 3 沒問題你可以保留
    g.load_ply(path)
    print(f"Loaded {g.get_xyz.shape[0]} gaussians from {path}")
    models.append(g)

for g, path in zip(models, ply_paths):
    g.load_ply(path)

json_paths = [p.replace(".ply", "_phys.json") for p in ply_paths]

# 呼叫合併儲存函式
save_combined_gaussians_to_ply_json_from_models(models, json_paths, filename_prefix="merge_dir/merged_gaussians")

g = GaussianModel(3)
g.load_ply("merge_dir/merged_gaussians.ply")
# g1 = GaussianModel(3)
# g1.load_ply("output_groups/sand.ply")
# show_3d_points(g1.get_xyz)
# compare_tensors("XYZ", g.get_xyz.cpu(), g1.get_xyz.cpu())
# compare_tensors("Features DC", g._features_dc.cpu(), g1._features_dc.cpu())
# compare_tensors("Features Rest", g._features_rest.cpu(), g1._features_rest.cpu())
# compare_tensors("Opacity", g._opacity.cpu(), g1._opacity.cpu())
# compare_tensors("Scaling", g._scaling.cpu(), g1._scaling.cpu())
# compare_tensors("Rotation", g._rotation.cpu(), g1._rotation.cpu())
with open("merge_dir/merged_gaussians_phys.json", "r") as f:
    phys = json.load(f)

material_list = phys["material"]
if len(material_list) != g.get_xyz.shape[0]:
    print(f"[❌] Mismatch: Gaussian count = {g.get_xyz.shape[0]}, Material count = {len(material_list)}")
else:
    print(f"[✅] Matched: {len(material_list)} gaussians and materials")
