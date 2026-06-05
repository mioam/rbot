from pathlib import Path

import numpy as np

from rbot.common.transformation import mat_to_xyz_rot, xyz_rot_to_mat


class Projector:
    def __init__(self, calib_path):
        calib_path = Path(calib_path)
        self.cam_to_base = {}
        files = calib_path.glob('*_camT.npy')
        print(calib_path)
        for file in files:
            cam_id = file.stem.split('_')[0]
            cam_to_base = np.load(file)
            # print(f'loaded cam {cam_id}', cam_to_base)
            self.cam_to_base[cam_id] = cam_to_base

        self.cam_intr = {}
        files = calib_path.glob('*_intr.npy')
        for file in files:
            cam_id = file.stem.split('_')[0]
            cam_intr = np.load(file)
            self.cam_intr[cam_id] = cam_intr

    def project_tcp_to_camera_coord(
        self, tcp, cam, rotation_rep='quaternion', rotation_rep_convention=None
    ):
        return mat_to_xyz_rot(
            np.linalg.inv(self.cam_to_base[cam])
            @ xyz_rot_to_mat(
                tcp,
                rotation_rep=rotation_rep,
                rotation_rep_convention=rotation_rep_convention,
            ),
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention,
        )

    def project_tcp_to_base_coord(
        self, tcp, cam, rotation_rep='quaternion', rotation_rep_convention=None
    ):
        return mat_to_xyz_rot(
            self.cam_to_base[cam]
            @ xyz_rot_to_mat(
                tcp,
                rotation_rep=rotation_rep,
                rotation_rep_convention=rotation_rep_convention,
            ),
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention,
        )

    def get_cam_to_base(self, cam):
        return self.cam_to_base[cam]

    def get_cam_intr(self, cam):
        return self.cam_intr[cam]
