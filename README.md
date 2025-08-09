# **AHMGaussian: Automatic Hybrid-Material Simulation with Gaussians**
## **Motivation**

> **Goal:** Generate **physics-informed** 3D objects from images, supporting **diverse materials**.
> **Base method:** *PhysGaussian* — extended to overcome its limitations.

**Key Drivers:**

* **Rich Material Support** – handle a wide range of materials in simulation.
* **Ease of Parameter Setup** – eliminate tedious manual tuning.
* **Visual Interactivity** – provide a user-friendly UI for adjusting physics parameters.
* **Hybrid Material Handling** – improve support for objects with multiple material types.
## **Contributions**

* **Auto Material Recognition** — *Recognize → Assign → Simulate*
  Automatically identify material types from input, assign physics parameters, and run simulation.

* **Config UI Tool** — *Visualize → Adjust → Export*
  Interactive Unity-based interface for real-time parameter tuning and export.

* **Blob-Level Hybrid Materials**
  Support objects composed of multiple distinct material regions.

* **Extended Material Library**
  Include more material categories for broader simulation coverage.


## Simulation Results
<table>
  <tr>
    <th style="text-align:center">With Hybrid Material</th>
    <th style="text-align:center">Without Hybrid Material</th>
  </tr>
  <tr>
    <td><img src="report\Hybrid1.gif" width="350"/></td>
    <td><img src="report\Hybrid2.gif" width="350"/></td>
  </tr>
  <tr>
    <th style="text-align:center">More Particle Filling</th>
    <th style="text-align:center">Less Particle Filling</th>
  </tr>
  <tr>
    <td><img src="report\Soil1.gif" width="350"/></td>
    <td><img src="report\Soil2.gif" width="350"/></td>
  </tr>
  <tr>
    <th style="text-align:center">Low Viscosity</th>
    <th style="text-align:center">High Viscosity</th>
  </tr>
  <tr>
    <td><img src="report\viscosity1.gif" width="350"/></td>
    <td><img src="report\viscosity2.gif" width="350"/></td>
  </tr>
  <tr>
    <th style="text-align:center">Low Amtitude of Disturbance</th>
    <th style="text-align:center">High Amtitude of Disturbance</th>
  </tr>
  <tr>
    <td><img src="report\water1.gif" width="350"/></td>
    <td><img src="report\water2.gif" width="350"/></td>
  </tr>
  <tr>
    <th style="text-align:center">Smaller than Yield_Stress</th>
    <th style="text-align:center">Larger than Yield_Stress</th>
  </tr>
  <tr>
    <td><img src="report\Metal1.gif" width="350"/></td>
    <td><img src="report\Metal2.gif" width="350"/></td>
  </tr>
  <tr>
    <th style="text-align:center">Smaller than Yield_Stress</th>
    <th style="text-align:center">Larger than Yield_Stress</th>
  </tr>
  <tr>
    <td><img src="report\elastic1.gif" width="350"/></td>
    <td><img src="report\elastic2.gif" width="350"/></td>
  </tr>
</table>

## **Environment Setup & Execute**

* It's recommended to first install dependencies from the original `physGaussian/requirements.txt`, and then manually install any remaining packages as needed (they're generally easy to resolve).
* Alternatively, you can install everything via `environment.yml` for convenience.
* [Pre-trained ficus model](https://drive.google.com/file/d/1G2HW4vT4hx6bkbWmWoy11JqtPmC5g26e/view?usp=sharing)
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
## Interactive Parameter Editing (Unity)

We provide an **[interactive Unity-based interface](https://github.com/r13944003/EV_Final_UnityGaussianSplatting.git)** (developed by [@FriendName](https://github.com/FriendUsername))  
to visualize simulation results and adjust **material parameters** or **simulation configurations** in real-time.

**How it works:**
1. Export grouped `.ply` files and their corresponding `parameter.json` from **AHMGaussian**.
2. Load them into the Unity interface.
3. Modify parameters such as:
   - Material stiffness (E, ν)
   - Density
   - Friction, cohesion
   - Boundary conditions
4. Save changes back to `.json`.
5. Run **AHMGaussian simulation** again with the updated parameters.

**Note:**  
Unity handles visualization and parameter editing.  
All physics simulation is still executed by the AHMGaussian Python pipeline.

**Post Overview:**




[![Poster Preview](report/poster1.jpg)](report/post.pdf)
[![Poster Preview](report/poster2.jpg)](report/post.pdf)

