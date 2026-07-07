from datetime import datetime
from pathlib import Path
import time

import numpy as np
from PIL import Image
import tyro
import yaml

from rbot.calib.utils import checkChessboard
from rbot.common.transformation import rot_mat_z_axis, rotation_transform
from rbot.device.camera import CameraD400
from rbot.device.robot import FlexivRobot
from rbot.utils.tools import imshow

ROBOT_IP = '192.168.2.100'
FPS = 30
# 244222073667
# [750612070265]

TARGET = np.array([0.4, 0.0, 0.0])


def look_at(camera_pos, target):
    """
    构造使TCP +Z指向target的旋转
    """
    z = target - camera_pos
    z = z / np.linalg.norm(z)
    y0 = np.array([0.0, 1.0, 0.0])
    x = np.cross(y0, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    Rmat = np.column_stack((x, y, z))
    return Rmat


def add_roll(R_base, roll_deg):
    """
    绕相机自身Z轴滚转
    """
    return R_base @ rot_mat_z_axis(roll_deg / 180.0 * np.pi)


def process(robot, cams, cams_dir, pose_dir, pose):
    robot.send_tcp_pose(pose, slow=True)

    curr_time = int(time.time() * 1000)

    cams_data = [cam.get_data() for cam in cams]
    tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
    N_CAM = len(cams)
    assert len(cams_dir) == N_CAM
    flags = []
    for i in range(N_CAM):
        flag, result_img = checkChessboard(cams_data[i][0])
        imshow(f'{i}', result_img)
        flags.append(flag)

        Image.fromarray(result_img).save(cams_dir[i] / f'{curr_time}r.png')
        Image.fromarray(cams_data[i][0]).save(cams_dir[i] / f'{curr_time}c.png')
        Image.fromarray(cams_data[i][1]).save(cams_dir[i] / f'{curr_time}d.png')
        np.savetxt(pose_dir / f'{curr_time}.txt', tcpPose)

    return flags


def main(cam_hand: str, cam_side: list[str], name: str = None):

    robot = FlexivRobot(ROBOT_IP)
    robot.send_tcp_pose((0.4, 0, 0.5, 0, 0, 1, 0))

    if name is None:
        name = datetime.now().strftime('%y%m%d')
    DIR = Path.home() / 'calib' / name
    cams_id = [cam_hand, *cam_side]
    cams = [CameraD400(c, fps=FPS) for c in cams_id]
    cams_dir = [DIR / f'{c}' for c in cams_id]
    pose_dir = DIR / 'pose'
    DIR.mkdir()
    pose_dir.mkdir()
    for d in cams_dir:
        d.mkdir()
    with (DIR / 'config.yaml').open('w') as f:
        yaml.dump(
            {
                'cam_hand': cam_hand,
                'cam_side': cam_side,
            },
            f,
        )
    for i in range(len(cams_id)):
        intr = cams[i].getIntrinsics()
        np.save(DIR / f'{cams_id[i]}_intr.npy', intr)

    poses = gen_pose()
    print(f'poses: {len(poses)}')
    for pose in poses:
        # robot.send_tcp_pose(pose, slow=True)
        flags = process(robot, cams, cams_dir, pose_dir, pose)
        print(flags)


def gen_pose():
    poses = []
    heights = [0.45, 0.55]
    radii = [0.1, 0.15]
    angles = np.arange(15, 360, 60)
    rolls = [20, 0, -20]

    for h in heights:
        for radius in radii:
            for angle_deg in angles:
                theta = np.deg2rad(angle_deg)

                x = TARGET[0] + radius * np.cos(theta)
                y = TARGET[1] + radius * np.sin(theta)
                z = h

                pos = np.array([x, y, z])

                R_base = look_at(pos, TARGET)

                for roll in rolls:
                    R_final = add_roll(R_base, roll)
                    quat = rotation_transform(
                        R_final,
                        from_rep='matrix',
                        to_rep='quaternion',
                    )

                    poses.append(np.concatenate([pos + R_final[0] * 0.05, quat]))
                    print(poses[-1])
    return poses


# python src/rbot/calib/collect_data2.py --cam-hand 244222073667 --cam-side 750612070265
if __name__ == '__main__':
    tyro.cli(main)
