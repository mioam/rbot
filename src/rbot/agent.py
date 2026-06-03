"""
Evaluation Agent.
"""

import time

import cv2
import numpy as np

from rbot.device.camera import CameraD400
from rbot.device.robot import FlexivGripper, FlexivRobot

# from rbot.utils.transformation import xyz_rot_transform

BLOCK_TIME = 0.1 - 0.03


class Agent:
    def __init__(
        self, robot_ip='192.168.2.100', camera_serials=('135122075425',), **kwargs
    ):
        print('Init robot, gripper, and camera.')
        self.robot = FlexivRobot(robot_ip)
        self.robot.send_tcp_pose(self.ready_pose, slow=True)
        time.sleep(2)
        self.gripper = FlexivGripper(self.robot)

        self.camera = [
            CameraD400(camera_serial, fps=30, res=(640, 480))
            for camera_serial in camera_serials
        ]
        self.camera_serials = camera_serials
        self.use_hand = False
        # self.use_hand = True
        if self.use_hand:
            self.camera_h = CameraD400('104122061850', res=(640, 480))
            self.intrinsics_h = self.camera_h.getIntrinsics()
        print('Init done')

    @property
    def ready_pose(self):
        return np.array([0.6, 0, 0.3, 0, 0.5**0.5, -(0.5**0.5), 0])

    def _get_observation(self, i):
        colors, depths = self.camera[i].get_data()
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
        return colors, depths / 1000.0

    def get_observation(self):
        return [self._get_observation(i) for i in range(len(self.camera))]

    def get_tcp_pose(self):
        tcp_pose = self.robot.get_tcp_pose()
        return tcp_pose

    def get_frame(self):
        cam_data = {}
        for camera in self.camera:
            color_image, depth_image = camera.get_data()
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cam_data[f'observation.images.{camera.serial}'] = color_image
            cam_data[f'observation.depths.{camera.serial}'] = depth_image
        tcpPose, jointPose, _, _ = self.robot.get_robot_state()
        gripperPose = self.gripper.get_gripper_state()
        frame = cam_data | {
            'observation.state.joint': np.array(jointPose).astype(np.float32),
            'observation.state.tcp': np.array(tcpPose).astype(np.float32),
            'observation.state.gripper': np.array([gripperPose]).astype(np.float32),
        }
        return frame

    def set_tcp_pose(
        self,
        pose,
        # rotation_rep,
        # rotation_rep_convention=None,
        blocking=False,
        slow=False,
    ):
        tcp_pose = pose
        self.robot.send_tcp_pose(tcp_pose, slow=slow)
        if blocking:
            time.sleep(BLOCK_TIME)

    def set_gripper_width(self, width, force=30, blocking=False):
        # width = 1000 if width > 500 else 0
        # print(width)
        self.gripper.move(width, force=force)
        self.last_width = width
        if blocking:
            time.sleep(2)

    def get_gripper_width(self):
        # WARNING: not realtime width
        return self.gripper.get_gripper_state()
        # if hasattr(self, 'last_width'):
        #     return self.last_width
        # return 1000

    def stop(self):
        self.robot.stop()


if __name__ == '__main__':
    agent = Agent(camera_serials=['750612070265'])
    obv = agent.get_observation()
    tcp = agent.get_tcp_pose()
    # agent.set_gripper_width()
    agent.set_tcp_pose(tcp, blocking=True)
    from rbot.utils.tools import show

    show({
        'obv': obv,
        'tcp': tcp,
    })
