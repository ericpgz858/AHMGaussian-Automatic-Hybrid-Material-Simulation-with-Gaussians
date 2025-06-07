import sys
sys.path.append("gaussian-splatting")
import os
import torch
import cv2
import argparse
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.decode_param import decode_param_json
from utils.transformation_utils import *
from utils.render_utils import *
from utils.graphics_utils import focal2fov
from gaussian_renderer import render
from diff_gaussian_rasterization import GaussianRasterizationSettings
from utils.sh_utils import eval_sh
from scene.cameras import Camera as GSCamera

class PipelineParamsNoparse:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False

def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply("merge_dir/merged_gaussians.ply")
    return gaussians



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Load config
    print("Loading scene config...")
    (
        _,
        _,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    print("Loading gaussians...")
    gaussians = load_checkpoint(args.model_path)
    pipeline = PipelineParamsNoparse()

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    print("Extracting parameters...")
    params = load_params_from_gs(gaussians, pipeline)

    pos = params["pos"]
    cov3D = params["cov3D_precomp"]
    opacity = params["opacity"]
    shs = params["shs"]
    screen_points = params["screen_points"]

    # Apply mask
    mask = opacity[:, 0] > preprocessing_params["opacity_threshold"]
    pos = pos[mask]
    cov3D = cov3D[mask]
    opacity = opacity[mask]
    shs = shs[mask]
    screen_points = screen_points[mask]

    print("Rendering frames...")
    height = width = None
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]

    camera_params = {
        "default_camera_index": 0,
        "show_hint": False,
        "init_azimuthm": 0.0,         # 初始方位角（水平旋轉角度）
        "init_elevation": 15.0,       # 初始仰角（上下觀看角度）
        "init_radius": 2.0,           # 與物體中心的距離
        "move_camera": True,          # 是否繞場景旋轉
        "delta_a": 2.0,               # 每幀增加的方位角度
        "delta_e": 0.0,               # 每幀仰角變化量（可設為 0）
        "delta_r": 0.0,               # 每幀距離變化量（可設為 0）
    }

    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            args.model_path,
            default_camera_index=camera_params["default_camera_index"],
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

        rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)

        rot = torch.eye(3, device="cuda").expand(pos.shape[0], 3, 3)
        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)

        rendering, _ = rasterize(
            means3D=pos,
            means2D=screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D,
        )

        image = rendering.permute(1, 2, 0).detach().cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if height is None or width is None:
            height = image.shape[0] // 2 * 2
            width = image.shape[1] // 2 * 2

        cv2.imwrite(os.path.join(args.output_path, f"{frame:04d}.png"), 255 * image)

    if args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )
