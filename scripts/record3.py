from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import tyro

from rbot.common.precise_sleep import precise_wait
from rbot.device.camera import CameraD400
from rbot.device.keyboard import KeyboardCounter
from rbot.device.robot import FlexivGripper, FlexivRobot
from rbot.device.sigma import Sigma7
from rbot.record import RawDataset
from rbot.utils.tools import imshow


class Controller:
    def __init__(self) -> None:
        self.keyboard = KeyboardCounter()
        self.quit = False
        self.reset()

    def reset(self):
        self.start = False
        self.detach = None
        self.discard = False
        self.finish = False

    def update(self):
        d = self.keyboard.get()
        if 'p' in d:
            if self.detach == 'detach':
                self.detach = 'resume'
            else:
                self.detach = 'detach'
        if 'i' in d:
            self.detach = 'init'
        # if 'r' in d:
        #     self.detach = 'resume'
        if 's' in d:
            self.start = True
        if 'f' in d:
            self.finish = True
        if 'q' in d:
            self.quit = True
        if 'd' in d:
            self.discard = True


def init(robot: FlexivRobot, sigma: Sigma7):
    robot.init_pose = np.array((0.5, 0, 0.4, 0, np.sin(0), np.cos(0), 0))
    robot.send_tcp_pose(robot.init_pose)
    sigma.detach_init()
    time.sleep(5)
    print('init done')


FPS = 10
# FPS_control = 30


def record(
    robot: FlexivRobot,
    gripper: FlexivGripper,
    cameras: list[CameraD400],
    sigma: Sigma7,
    controller: Controller,
    dataset: RawDataset,
    task: str,
):
    dataset.new_demo()
    # print(camera.getIntrinsics())
    # color_image, depth_image = camera.get_data()
    # color_image: 460,640,3  0~255
    # depth_image: 460,640  0~4681 [mm]

    controller.reset()

    while True:
        curr_time = int(time.time() * 1000)
        frame_start = time.monotonic()

        controller.update()
        if controller.detach:
            # print(controller.detach)
            if controller.detach == 'detach':
                sigma.detach()
            elif controller.detach == 'init':
                sigma.detach()
                init(robot, sigma)
                controller.detach = 'detach'
            elif controller.detach == 'resume':
                sigma.resume()
                controller.detach = None
        elif controller.quit or controller.discard or controller.finish:
            break
        else:
            cam_data = {}
            for camera in cameras:
                color_image, depth_image = camera.get_data()
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                cam_data[f'observation.images.{camera.serial}'] = color_image
                cam_data[f'observation.depths.{camera.serial}'] = depth_image

            tcpPose, jointPose, _, _ = robot.get_robot_state()
            gripperPose = gripper.get_gripper_state()
            diff_p, diff_r, width = sigma.get_control()  # ~0.01s for rpc
            diff_p = diff_p + robot.init_pose[:3]
            diff_p = np.clip(diff_p, [0.1, -0.5, 0.135], [0.9, 0.5, 0.5])  # for safty
            diff_r = diff_r * R.from_quat(robot.init_pose[3:], scalar_first=True)
            tcp_action = np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0)
            # Send command.
            robot.send_tcp_pose(tcp_action)
            gripper.move(width)

            for camera in cameras:
                vis = cam_data[f'observation.images.{camera.serial}']
                h, w, _ = vis.shape
                # midcrop
                minl = min(w, h)

                cv2.rectangle(
                    vis,
                    ((w - minl) // 2, (h - minl) // 2),
                    ((w + minl) // 2, (h + minl) // 2),
                    (255, 0, 0) if not controller.start else (0, 255, 0),
                    2,
                )

                # 写文字
                cv2.putText(
                    vis,
                    f'{controller.start}',
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                imshow(f'{camera.serial}', vis)

            if controller.start:
                action = np.append(tcp_action, width)
                frame = cam_data | {
                    'observation.state.joint': np.array(jointPose).astype(np.float32),
                    'observation.state.tcp': np.array(tcpPose).astype(np.float32),
                    'observation.state.gripper': np.array([gripperPose]).astype(
                        np.float32
                    ),
                    'actions': action.astype(np.float32),
                    'task': task,
                    'curr_time': curr_time,
                }

                dataset.add_frame(frame)

        precise_wait(frame_start + 1.0 / FPS)

    if not controller.start or controller.quit or controller.discard:
        print('discard')
        dataset.discard()
        return
    dataset.save()


def main(
    camera_serials=['750612070265', '244222073667'],
):
    cprint(f'{camera_serials=}', 'red')

    robot = FlexivRobot('192.168.2.100')
    gripper = FlexivGripper(robot)
    cameras = [CameraD400(s) for s in camera_serials]
    sigma = Sigma7(pos_scale=4)
    # sigma = Sigma7RPC()
    controller = Controller()
    dataset = RawDataset(
        Path('/ssd1/mzc/data')
        / f'raw/record-{datetime.now().strftime("%Y%m%d-%H:%M:%S")}'
    )

    while not controller.quit:
        record(
            robot,
            gripper,
            cameras,
            sigma,
            controller,
            dataset,
            task='',
        )
    print('quit')


if __name__ == '__main__':
    tyro.cli(main)
