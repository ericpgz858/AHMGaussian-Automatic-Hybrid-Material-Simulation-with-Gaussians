# PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
## My Update
### preprocess.py

| 參數名稱              | 說明                                                               |
| ----------------- | ---------------------------------------------------------------- |
| `--model_path`    | 輸入的 Gaussian 模型目錄，需包含 `point_cloud/iteration_xx/point_cloud.ply` |
| `--output_path`   | 輸出分群 `.ply` 與對應 `.json` 的資料夾路徑(如果不是"output_groups", merge_gaussian.py 裡面在 load 的時候要改變 path)                                   |
| `--camera_config` | NeRF 格式的相機設定（例如 `transforms_train.json`）                         |
| `--camera_list`   | 這部分需要是 julia 的 mask folder，我在跟她確認格式                                      |

**目前是指透過一張 segment mask 做分群，把 seg_map.png 放在與 preprocess.py 同一層資料夾下面，只有做分群沒有 filling, transorm 等其他的 preprocess**
```shell
python preprocess.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path output_groups \
  --camera_config ../nerf_synthetic/ficus/transforms_train.json \
  --camera_list ../nerf_synthetic/ficus/
```
### merge_gaussian.py
* 預設將 output_groups 底下的 ply 與 json 合併成一個 gaussian ply file 與 json file
* 預設存在 merge_dir 下面 
```shell
python merge_gaussians.py 
```
###  my_simulation.py
| 參數名稱               | 說明                                    |
| ------------------ | ------------------------------------- |
| `--model_path`     | 原始 Gaussian 模型所在資料夾（get_camera_view 需要讀取 camera data 來做 render 時的 camera 參數） |
| `--merge_gaussian` | 合併後的 `.ply` 模型，包含所有高斯點（沒做 filling）           |
| `--phys_config`    | 對應 `.ply` 的物理參數 JSON，例如 `*_phys.json` |
| `--config`         | 模擬與渲染的全域設定檔（包含時間、邊界條件、材質設定等）          |
| `--output_path`    | 渲染圖片與影片的輸出資料夾                         |
| `--render_img`     | 是否輸出每一幀的渲染影像                          |
| `--compile_video`  | 是否將渲染的影像合成為影片                         |
| `--white_bg`       | 是否啟用白色背景（預設為黑色）                       |

```bash
python my_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --merge_gaussian merge_dir/merged_gaussians.ply \
  --output_path test_output \
  --config ./config/my_ficus.json \
  --phys_config merge_dir/merged_gaussians_phys.json \
  --render_img \
  --compile_video \
  --white_bg
```
### [[Project Page](https://xpandora.github.io/PhysGaussian/)] [[arXiv](https://arxiv.org/abs/2311.12198)] [[Video](https://www.youtube.com/watch?v=V96GfcMUH2Q)]

Tianyi Xie<sup>1</sup>\*, Zeshun Zong<sup>1</sup>\*, Yuxing Qiu<sup>1</sup>\*, Xuan Li<sup>1</sup>\*, Yutao Feng<sup>2,3</sup>, Yin Yang<sup>3</sup>, Chenfanfu Jiang<sup>1</sup><br>
<sup>1</sup>University of California, Los Angeles, <sup>2</sup>Zhejiang University, <sup>3</sup>University of Utah <br>
*Equal contributions

![teaser-1.jpg](_resources/teaser-1.jpg)

Abstract: *We introduce PhysGaussian, a new method that seamlessly integrates physically grounded Newtonian dynamics within 3D Gaussians to achieve high-quality novel motion synthesis. Employing a customized Material Point Method (MPM), our approach enriches 3D Gaussian kernels with physically meaningful kinematic deformation and mechanical stress attributes, all evolved in line with continuum mechanics principles. A defining characteristic of our method is the seamless integration between physical simulation and visual rendering: both components utilize the same 3D Gaussian kernels as their discrete representations. This negates the necessity for triangle/tetrahedron meshing, marching cubes, ''cage meshes,'' or any other geometry embedding, highlighting the principle of ''what you see is what you simulate (WS2).'' Our method demonstrates exceptional versatility across a wide variety of materials--including elastic entities, plastic metals, non-Newtonian fluids, and granular materials--showcasing its strong capabilities in creating diverse visual content with novel viewpoints and movements.*

## News
- [2024-03-27] Release a Colab notebook for quick start.[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/165WAoLw2HK4WifsA4Ngqgeke6gJVQXDm?usp=sharing)
- [2024-03-03] Code Release.
- [2024-02-27] Our paper has been accpetd by CVPR 2024!
- [2023-12-20] Our [MPM solver code](https://github.com/zeshunzong/warp-mpm) is open sourced!

## Cloning the Repository
This repository uses original gaussian-splatting as a submodule. Use the following command to clone:

```shell
git clone --recurse-submodules git@github.com:XPandora/PhysGaussian.git
```

## Setup

### Python Environment
To prepare the Python environment needed to run PhysGaussian, execute the following commands:
```shell
conda create -n PhysGaussian python=3.9
conda activate PhysGaussian

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```
By default, We use pytorch=2.0.0+cu118.
### Quick Start
We provide several pretrained [Gaussian Splatting models](https://drive.google.com/drive/folders/1EMUOJbyJ2QdeUz8GpPrLEyN4LBvCO3Nx?usp=drive_link) and their corresponding `.json` config files in the `config` directory.

To simulate a reconstructed 3D Gaussian Splatting scene, run the following command:
```shell
python gs_simulation.py --model_path <path to gs model> --output_path <path to output folder> --config <path to json config file> --render_img --compile_video
```
The images and video results will be saved to the specified output_path.

If you want a quick try, run:
```shell
pip install gdown
bash download_sample_model.sh
python gs_simulation.py --model_path ./model/ficus_whitebg-trained/ --output_path output --config ./config/ficus_config.json --render_img --compile_video --white_bg
```
Hopefully, you will see a video result like this:

<img src="./demo/ficus.gif" width="300"/>

## Custom Dynamics
To generate custom dynamics, follow these guidelines:

### Gaussian Splatting Reconstruction
Begin by reconstructing a 3D GS scene as per [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

### Data Preprocessing
Before simulating Gaussian kernels as continuum particles, perform the following preprocessing steps:
1. Remove Gaussian kernels with low opacity.
2. Rotate the 3D scene to make it align with the coordinate plane (e.g., bottom surface parallel to the xy plane).
3. Define a cuboid simulation area.
4. Center and scale the simulation area within a unit cube.
5. Optionally, fill internal voids with particles.

Related parameters, such as rotation axis and degree, should be provided in the config file. For [Nerf Synthetic Dataset](https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=drive_link), the reconstructed results typically already align with the axis.  For custom datasets, we use 3D software, e.g. [Houdini](https://www.sidefx.com/), to view the distribution of the Gaussian kernels and determine how to rotate and select the scene for simulation readiness.

### Config Json File
A single `.json` file should detail all data preprocessing and simulation parameters for each scene. Key parameters include:

- Data Preprocessing Parameters:
    - `opacity_threshold`: Filters out Gaussian kernels with opacity below this threshold.
    - `rotation_degree (list)` and `rotation_axis (list)`: Rotate the scene to align with the grid.
    - `sim_area (list)`: Choose the particles within a bounding box for simulation. The expected format is `[xmin, xmax, ymin, ymax, zmin, zmax]`.
    - `particle_filling (dict)`: Specify a cubic area to fill internal particles. Tuning ```density_threshold``` and ```search_threshold``` is usually needed for optimal filling results. See more details below.
- Simulation Parameters:
    - `material`: Available material types include `jelly`, `metal`, `sand`, `foam`, `snow` and `plasticine`.
    - `E`: Young's modulus 
    - `nu`: Poisson's Ratio
    - `density`: Material density  
    - `g`: Gravity
    - `substep_dt`: Simulation time step size 
    - `n_grid`: MPM grid size
    - `boundary_conditions (list)`: Boundary conditions can be enforced on either particles or grids, allowing manipulation of Gaussian kernels via external forces.
- Export Parameters:
    - `frame_dt`: Duration of each frame
    - `frame_num`: Total number of frames to export
    - `default_camera_index`: Camera view index from the training set

Please see sample config files under the `config` folder for reference. 

#### Particle Filling
Optionally, we employ a ray-collision-based method to detect inner grids for particle filling. To use this, specify the following parameters:

- `n_grid`: Particle filling grid size.
- `density_threshold`: Grid cells with density above this threshold will be treated as part of the surface shell.
- `search_exclude_direction`: A parameter (list of ints) for internal filling condition 1 in PhysGaussian. We won't cast rays in these excluded directions. The mapping between ints and directions is: 0, 1, 2, 3, 4, 5 (+x, -x, +y, -y, +z, -z).
- `ray_cast_direction`: tA parameter for internal filling condition 2 in PhysGaussian. Along this direction, we will detect the number of collision times. The mapping between ints and directions is the same as `search_exclude_direction`.
- `max_particles_per cell`: The number of particles to fill for each grid cell.
- `boundary`: Specify a well-reconstructed cubic area to perform particle filling.

Note: This particle filling algorithm is sensitive to Gaussian kernel distribution and may produce unsatisfying filling results if Gaussians are too noisy.

#### Boundary Condition
To fix or move the reconstructed object, specify the boundary condition either on grids or particles. Some commonly used boundary condition types include:

- `bounding_box`: Prevents particles from moving outside the MPM simulation area.
- `cuboid`: Enforces a boundary condition on the grid. Also specify other necessary parameters:
    - `point`: Center of the cubic area, e.g. `[1, 1, 1]`
    - `size`: Size of the cubic area (half of the width, height and depth), e.g. `[0.2, 0.2, 0.2]`
    - `vecloticy`: Velocity assigned to the grids, e.g. `[0, 0, 0]`
    - `start_time` and `end_time`: Time duration of this boundary condition
- `enforce_particle_translation`: Enforces a boundary condition on particles with parameters similar to those for grids.

## Citation

```
@article{xie2023physgaussian,
      title={PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics}, 
      author={Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
      journal={arXiv preprint arXiv:2311.12198},
      year={2023},
}
```
