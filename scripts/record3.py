from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import tyro

from rbot.agent import Agent
from rbot.common.image_util import Cropper
from rbot.common.precise_sleep import precise_wait
from rbot.device.keyboard import KeyboardCounter
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


def init(agent: Agent, sigma: Sigma7):
    agent.robot.init_pose = np.array((0.5, 0, 0.4, 0, np.sin(0), np.cos(0), 0))
    agent.robot.send_tcp_pose(agent.robot.init_pose, slow=True)
    sigma.detach_init()
    # time.sleep(5)
    print('init done')


FPS = 10
# FPS_control = 30
# MIN_Z = 0.135
MIN_Z = 0.10


def record(
    agent: Agent,
    sigma: Sigma7,
    controller: Controller,
    dataset: RawDataset,
    task: str,
    reference=None,
):
    dataset.new_demo()
    # color_image, depth_image = camera.get_data()
    # color_image: 460,640,3  0~255
    # depth_image: 460,640  0~4681 [mm]

    controller.reset()
    controller.detach = 'init'
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
                init(agent, sigma)
                controller.detach = 'detach'
            elif controller.detach == 'resume':
                sigma.resume()
                controller.detach = None
        elif controller.quit or controller.discard or controller.finish:
            break
        else:
            frame = agent.get_frame()

            diff_p, diff_r, width = sigma.get_control()  # ~0.01s for rpc
            diff_p = diff_p + agent.robot.init_pose[:3]
            diff_p = np.clip(diff_p, [0.1, -0.5, MIN_Z], [0.9, 0.5, 0.5])  # for safty
            diff_r = diff_r * R.from_quat(agent.robot.init_pose[3:], scalar_first=True)
            tcp_action = np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0)
            # Send command.
            agent.set_tcp_pose(tcp_action)
            agent.set_gripper_width(width)

            for i, camera in enumerate(agent.camera):
                vis = frame[f'observation.images.{camera.serial}'].copy()
                h, w, _ = vis.shape
                # midcrop
                minl = min(w, h)

                if i == 0:
                    cv2.rectangle(
                        vis,
                        ((w - minl) // 2, (h - minl) // 2),
                        ((w + minl) // 2, (h + minl) // 2),
                        color=(255, 0, 0) if not controller.start else (0, 255, 0),
                        thickness=2,
                    )
                else:
                    cv2.rectangle(
                        vis,
                        ((w - minl), (h - minl) // 2),
                        ((w), (h + minl) // 2),
                        color=(255, 0, 0) if not controller.start else (0, 255, 0),
                        thickness=2,
                    )

                # 写文字
                cv2.putText(
                    vis,
                    f'{controller.start}',
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                imshow(f'{camera.serial}', vis)

            if controller.start:
                action = np.append(tcp_action, width)
                frame = frame | {
                    'actions': action.astype(np.float32),
                    'task': task,
                    'curr_time': curr_time,
                }

                dataset.add_frame(frame)

        precise_wait(frame_start + 1.0 / FPS)

    if not controller.start or controller.quit or controller.discard:
        print('discard')
        dataset.discard()
        return False
    dataset.save()
    return True


def main(
    camera_serials=['750612070265', '244222073667'],
    ref: str | None = None,
    start: int = 0,
):
    cprint(f'{camera_serials=}', 'red')

    agent = Agent(camera_serials=camera_serials)
    sigma = Sigma7(pos_scale=4)
    # sigma = Sigma7RPC()
    controller = Controller()
    dataset = RawDataset(
        Path('/ssd1/mzc/data')
        / f'raw/record-{datetime.now().strftime("%Y%m%d-%H:%M:%S")}'
    )
    if ref is not None:
        ref = Path(ref)
        cnt = start
        cropper = Cropper()
        # samples = np.loadtxt(ref / 'samples.txt').reshape(-1, 6)

    while not controller.quit:
        if ref is not None:
            controller.reset()
            cprint(f'init with {cnt}, press f to finish', 'green')
            key = 'observation.images.750612070265'
            ref_img = np.array(Image.open(ref / key / f'{cnt:05}.png'))
            while not controller.finish:
                controller.update()
                frame = agent.get_frame()
                cropper.crop_frame(frame)
                imshow('init', frame[key] // 2 + ref_img // 2)
                time.sleep(0.1)
            cprint('initialized', 'green')
        flag = record(
            agent,
            sigma,
            controller,
            dataset,
            task='',
        )
        if ref is not None and flag:
            cprint(f'saved with {cnt}', 'green')
            cnt += 1
    print('quit')


if __name__ == '__main__':
    tyro.cli(main)
