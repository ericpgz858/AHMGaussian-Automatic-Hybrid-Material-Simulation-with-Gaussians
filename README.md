Here's a polished English version of your documentation with clearer structure and smoother phrasing:

---

# **AHMGaussian: Automatic Hybrid-Material Simulation with Gaussians**

## **My Update**

### `preprocess.py`

| Argument          | Description                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------- |
| `--model_path`    | Input Gaussian model directory. It must include `point_cloud/iteration_xx/point_cloud.ply`. |
| `--output_path`   | Output directory to save the grouped `.ply` files and their corresponding material `.json`. |
| `--camera_config` | NeRF-style camera configuration (e.g., `transforms_train.json`).                            |
| `--camera_list`   | Should point to a folder of masks (currently checking the format with Julia).               |

> Currently, this script performs segmentation using a single segmentation mask (`seg_map.png`) placed in the same directory as `preprocess.py`.
> It only performs material grouping — no filling, transformation, or other preprocessing steps are included yet.

```bash
python preprocess.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path output_groups \
  --camera_config ../nerf_synthetic/ficus/transforms_train.json \
  --camera_list ../nerf_synthetic/ficus/
```

---

### `my_simulation.py`

| Argument          | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `--model_path`    | Path to the original Gaussian model. Camera data is required for rendering.    |
| `--merge_folder`  | Folder containing grouped `.ply` models and their corresponding material `.json` from `preprocess.py` or [unity](https://github.com/r13944003/EV_Final_UnityGaussianSplatting.git).                  |
| `--config`        | Global configuration file for simulation and rendering (time, material, etc.). |
| `--output_path`   | Output folder for rendered images and video.                                   |
| `--render_img`    | Whether to output individual rendered frames.                                  |
| `--compile_video` | Whether to compile rendered frames into a video.                               |
| `--white_bg`      | Whether to render with a white background (default is black).                  |

```bash
python my_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --merge_folder output_groups \
  --output_path test_output \
  --config ./config/myficus_config_sand.json \
  --render_img \
  --compile_video \
  --white_bg
```

---

### **Environment Setup**

* It's recommended to first install dependencies from the original `physGaussian/requirements.txt`, and then manually install any remaining packages as needed (they're generally easy to resolve).
* Alternatively, you can install everything via `environment.yml` for convenience.
* [Pre-trained ficus model](https://drive.google.com/file/d/1G2HW4vT4hx6bkbWmWoy11JqtPmC5g26e/view?usp=sharing)
