from pathlib import Path

import cv2
import numpy as np


class Projector:
    def __init__(self, calib_path):
        calib_path = Path(calib_path)
        self.cam_to_base = {}
        files = calib_path.glob("*_camT.npy")
        print(calib_path)
        for file in files:
            cam_id = file.stem.split("_")[0]
            cam_to_base = np.load(file)
            # print(f'loaded cam {cam_id}', cam_to_base)
            self.cam_to_base[cam_id] = cam_to_base

        self.cam_intr = {}
        files = calib_path.glob("*_intr.npy")
        for file in files:
            cam_id = file.stem.split("_")[0]
            cam_intr = np.load(file)
            self.cam_intr[cam_id] = cam_intr

    def get_cam_to_base(self, cam):
        return self.cam_to_base[cam]

    def get_cam_intr(self, cam):
        return self.cam_intr[cam]


def _inpaint(img, missing_value=0):
    """
    pip opencv-python == 3.4.8.29
    :param image:
    :param roi: [x0,y0,x1,y1]
    :param missing_value:
    :return:
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    if scale < 1e-3:
        pdb.set_trace()
    # Has to be float32, 64 not supported.
    img = img.astype(np.float32) / scale
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale
    return img


def getXYZRGB(color, depth, robot_pose, camee_pose, camIntrinsics, inpaint=False):
    """
    Generate XYZRGB point cloud from RGB-D and poses.
        color: (H, W, 3)
        depth: (H, W)
    Returns:
        points: (N, 6) array [X, Y, Z, R, G, B]
    """

    H, W = depth.shape[:2]

    fx, fy, cx, cy = (
        camIntrinsics[0, 0],
        camIntrinsics[1, 1],
        camIntrinsics[0, 2],
        camIntrinsics[1, 2],
    )

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float32)

    # optional depth cleanup
    if inpaint:
        depth = _inpaint(depth)
    valid = z > 0.2

    z = z[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N,4)
    T = robot_pose @ camee_pose

    pts_world = (T @ pts_cam.T).T
    pts_world = pts_world[:, :3]
    rgb = color[valid].astype(np.float32)

    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    xyzrgb = np.concatenate([pts_world, rgb], axis=1)

    return xyzrgb
