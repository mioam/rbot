from pathlib import Path

import numpy as np
from PIL import Image
import rerun as rr
import tyro
import yaml

from rbot.calib.utils import getXYZRGB
from rbot.common.projector import Projector
from rbot.common.transformation import xyz_rot_to_mat

WORKSPACE_MIN = np.array([0.0, -0.5, -0.2])
WORKSPACE_MAX = np.array([1.0, 0.5, 1.0])


def vis(DIR, cam_id, timestamp, current_pose, camT, intr):
    color_filename = DIR / f'{cam_id}' / f'{timestamp}c.png'
    depth_filename = DIR / f'{cam_id}' / f'{timestamp}d.png'

    color = np.array(Image.open(color_filename))
    depth = np.array(Image.open(depth_filename)) / 1000.0

    cloud = getXYZRGB(
        color,
        depth,
        current_pose,
        camT,
        intr,
    )
    cloud = cloud[
        np.all((cloud[:, :3] > WORKSPACE_MIN) & (cloud[:, :3] < WORKSPACE_MAX), axis=1)
    ]
    rr.log(f'{cam_id}', rr.Points3D(cloud[:, :3], colors=cloud[:, 3:]))


def main(DIR: Path):
    rr.init('calib', spawn=True)

    with (DIR / 'config.yaml').open('r') as f:
        conf = yaml.load(f, yaml.BaseLoader)
    print(conf)
    cam_hand = conf['cam_hand']
    cam_sides = conf['cam_side']
    timestamps = [p.stem for p in (DIR / 'pose').glob('*.txt')]
    timestamps.sort()
    print(len(timestamps))
    projector = Projector(DIR / 'calib')

    # cam_hand_intr = np.load(DIR / 'calib' / f'{cam_hand}_intr.npy')
    # cam_hand_camT = np.load(DIR / 'calib' / f'{cam_hand}_camT.npy')

    for i, timestamp in enumerate(timestamps):
        pose_filename = DIR / 'pose' / f'{timestamp}.txt'
        ee_pose_tq = np.loadtxt(pose_filename).tolist()
        current_pose = xyz_rot_to_mat(ee_pose_tq, from_rep='quaternion')

        vis(
            DIR,
            cam_hand,
            timestamp,
            current_pose,
            projector.cam_to_base[cam_hand],
            projector.cam_intr[cam_hand],
        )
        for cam_side in cam_sides:
            if cam_side in projector.cam_to_base:
                vis(
                    DIR,
                    cam_side,
                    timestamp,
                    np.eye(4),
                    projector.cam_to_base[cam_side],
                    projector.cam_intr[cam_side],
                )
        rr.log(
            'hand/box',
            rr.Boxes3D(
                half_sizes=[0.5, 0.5, 0.25],
                centers=[0.5, 0, 0.25],
            ),
        )


if __name__ == '__main__':
    tyro.cli(main)
