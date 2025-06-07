import numpy as np
import json
import cv2
import open3d as o3d

def load_gaussians_from_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)  # shape: (N, 3)

def conver_to_opcv(c2w):
    """Convert camera to OpenCV format."""
    diag = np.diag([1, 1, -1, 1])
    return c2w @ diag

def load_camera_data(json_path):
    with open(json_path, 'r') as f:
        meta = json.load(f)

    H, W = 800, 800
    camera_angle_x = meta['camera_angle_x']
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    cx, cy = W / 2, H / 2

    frame = meta['frames'][1]
    print("frame data:", frame)
    c2w = np.array(frame['transform_matrix'])
    c2w = conver_to_opcv(c2w)  # Convert to OpenCV format
    w2c = np.linalg.inv(c2w)

    image_path = "../../nerf_synthetic/ficus/"+frame['file_path'] + ".png"
    return w2c, focal, cx, cy, H, W, image_path, c2w

def check_gaussian_space(gaussians, c2w):
    cam_pos = c2w[:3, 3]
    distances = np.linalg.norm(gaussians - cam_pos[None, :], axis=1)
    print("Average distance to camera center:", np.mean(distances))
    if 1.0 < np.mean(distances) < 5.0:
        print("Likely in world space.")
    else:
        print("Might be in camera space or wrong coordinates.")

def project_point(X_world, c2w, focal, cx, cy):
    R = c2w[:3, :3]
    T = c2w[:3, 3]

    X_cam = R.T @ (X_world - T)

    print("Z in camera space:", X_cam[2])  # 看是不是正常的正值
    if X_cam[2] <= 0:
        return None
    x = (X_cam[0] / X_cam[2]) * focal + cx
    y = -(X_cam[1] / X_cam[2]) * focal + cy
    return np.array([x, y])

def draw_projection(image_path, projected_points, H, W):
    img = cv2.imread(image_path)
    print(f"projecting {len(projected_points)} points onto the image.")
    for uv in projected_points:
        u, v = int(uv[0]), int(uv[1])
        print(f"Projecting point at ({u}, {v})")
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(img, (u, v), 2, (0, 255, 0), -1)
    cv2.imwrite("projected_output.png", img)
    print("Saved projection to 'projected_output.png'.")

def main():
    ply_path = "point_cloud.ply"
    json_path = "../../nerf_synthetic/ficus/transforms_train.json"

    gaussians = load_gaussians_from_ply(ply_path)
    w2c, focal, cx, cy, H, W, image_path, c2w = load_camera_data(json_path)

    check_gaussian_space(gaussians, c2w)

    projected_pixels = []
    for g in gaussians:
        uv = project_point(g, c2w, focal, cx, cy)
        # print(f"Projected point: {uv}")
        if uv is not None:
            projected_pixels.append(uv)

    draw_projection(image_path, projected_pixels, H, W)

if __name__ == "__main__":
    main()
