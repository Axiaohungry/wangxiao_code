# colmap2transforms.py (修复版)
import os
import json
import numpy as np
import struct
import argparse


# ==================== COLMAP Binary Reader Helpers ====================
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            # 修复点：这里应该是 iiQQ (4+4+8+8 = 24 bytes)
            params = read_next_bytes(fid, 24, "iiQQ")
            camera_id = params[0]
            model_id = params[1]
            width = params[2]
            height = params[3]

            model_name = {1: "PINHOLE", 2: "RADIAL", 4: "OPENCV"}.get(model_id, "UNKNOWN")

            # Pinhole params: fx, fy, cx, cy
            # Simple Radial: f, cx, cy, k
            num_params_map = {1: 4, 2: 5, 4: 8}
            num_params = num_params_map.get(model_id, 0)

            # 如果模型ID未识别，尝试根据剩余字节判断（通常是 PINHOLE 4个参数）
            if num_params == 0:
                num_params = 4

            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = (width, height, np.array(params))
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            # Skip 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)

            images[image_id] = (qvec, tvec, camera_id, image_name)
    return images


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


# ==================== Main Logic ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COLMAP binary to transforms.json")
    parser.add_argument("--source", "-s", required=True, help="Path to the dataset folder (should contain sparse/0)")
    args = parser.parse_args()

    # Paths
    sparse_path = os.path.join(args.source, "sparse", "0")
    cameras_bin = os.path.join(sparse_path, "cameras.bin")
    images_bin = os.path.join(sparse_path, "images.bin")

    if not os.path.exists(cameras_bin) or not os.path.exists(images_bin):
        print(f"Error: COLMAP binary files not found in {sparse_path}")
        exit(1)

    print(f"Reading cameras from {cameras_bin}...")
    cameras = read_cameras_binary(cameras_bin)
    print(f"Reading images from {images_bin}...")
    images = read_images_binary(images_bin)

    if not cameras or not images:
        print("Error: No cameras or images found in binary files.")
        exit(1)

    # Assuming all images share the same camera params usually
    cam_id = list(cameras.keys())[0]
    W, H, params = cameras[cam_id]
    fl_x = params[0]
    fl_y = params[1]

    # Calculate field of view
    angle_x = 2 * np.arctan(W / (2 * fl_x))
    angle_y = 2 * np.arctan(H / (2 * fl_y))

    out = {
        "camera_angle_x": float(angle_x),
        "camera_angle_y": float(angle_y),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "k1": 0, "k2": 0, "p1": 0, "p2": 0,
        "cx": float(params[2]),
        "cy": float(params[3]),
        "w": int(W),
        "h": int(H),
        "frames": []
    }

    print(f"Converting poses for {len(images)} images...")

    # Flip Y and Z axes for NeRF coordinate system
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for img_id in sorted(images.keys()):
        qvec, tvec, cam_id, name = images[img_id]

        # World-to-Camera rotation
        R = qvec2rotmat(qvec)
        # World-to-Camera translation
        t = tvec.reshape(3, 1)

        # Form 4x4 matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t.squeeze()

        # Invert to get Camera-to-World (c2w)
        c2w = np.linalg.inv(w2c)

        # Apply coordinate system flip
        c2w = np.matmul(c2w, flip_mat)

        frame = {
            "file_path": f"./images/{name}",
            "transform_matrix": c2w.tolist()
        }
        out["frames"].append(frame)

    # Save
    out_path = os.path.join(args.source, "transforms.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"Done! Saved to {out_path}")