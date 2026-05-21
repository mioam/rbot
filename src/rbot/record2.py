import json
import multiprocessing
import os
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

from rbot.device.camera import CameraD400
from rbot.device.keyboard import RecordKeyboard as Keyboard
from rbot.device.robot import FlexivGripper, FlexivRobot
from rbot.device.sigma import Sigma7

camera_serial = ['750612070265', '244222073667']
# camera_serial = ['135122075425', '104422070042']
calib_file = '0123'
reference = None
cprint(f'{camera_serial=}, {calib_file=}', 'red')

queue = multiprocessing.Queue()


def init_image(cam):

    while True:
        color, depth = cam.get_data()
        vis = color * 0.5 + reference * 0.5 if reference is not None else color
        vis = vis / 255.0
        cv2.imshow('a', vis)
        if cv2.waitKey(1):
            break


def save_data(
    color_image,
    depth_image,
    color_dir,
    depth_dir,
):
    # print(color_dir, depth_dir)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_image).save(color_dir)
    Image.fromarray(depth_image).save(depth_dir)
    # print('done')
    queue.get()


def init(robot: FlexivRobot, sigma: Sigma7):
    (x, y, z) = np.random.random(3) - 0.5
    x *= 0.1
    y *= 0.1
    z *= 0.05
    theta = (np.random.random() - 0.5) * np.pi / 2
    # print(x, y, z, theta)
    # robot.init_pose = (0.5+x, 0+y, 0.3+z, 0, np.sin(theta), np.cos(theta), 0)
    robot.init_pose = (0.5, 0, 0.4, 0, np.sin(0), np.cos(0), 0)

    robot.send_tcp_pose(robot.init_pose)
    sigma.detach_init()
    time.sleep(5)


CNT = 0  # 用来计数


def record(
    robot: FlexivRobot,
    gripper: FlexivGripper,
    cameras: list[CameraD400],
    sigma: Sigma7,
    keyboard: Keyboard,
    pool,
):
    start_time = int(time.time() * 1000)  # 想要统一需要*1000然后取整。但是应该不关键

    demo_path = os.path.join('/ssd1/mzc/data/record2', f'{start_time}')
    os.makedirs(demo_path)
    # print(camera.getIntrinsics())
    # color_image, depth_image = camera.get_data()
    # color_image: 460,640,3  0~255
    # depth_image: 460,640  0~4681 [mm]

    cam_path = [os.path.join(demo_path, f'cam_{s}') for s in camera_serial]
    color_dir = [os.path.join(path, 'color') for path in cam_path]
    depth_dir = [os.path.join(path, 'depth') for path in cam_path]
    for path in cam_path:
        os.mkdir(path)
    for path in color_dir:
        os.mkdir(path)
    for path in depth_dir:
        os.mkdir(path)

    tcp_dir = os.path.join(demo_path, 'tcp')
    joint_dir = os.path.join(demo_path, 'joint')
    action_dir = os.path.join(demo_path, 'action')
    gripper_dir = os.path.join(demo_path, 'gripper_command')

    os.mkdir(tcp_dir)
    os.mkdir(joint_dir)
    os.mkdir(gripper_dir)
    os.mkdir(action_dir)

    with open(os.path.join(demo_path, 'timestamp.txt'), 'w') as f:
        f.write(calib_file)

    keyboard.start = False
    keyboard.discard = False
    keyboard.finish = False
    cnt = 0
    start_time = time.time()
    last_time = None
    total_time = []
    while not keyboard.quit and not keyboard.discard and not keyboard.finish:
        # time.sleep(0.1)
        curr_time = int(time.time() * 1000)
        cam_data = []
        for camera in cameras:
            color_image, depth_image = camera.get_data()
            cam_data.append((color_image, depth_image))
        if keyboard.detach or keyboard.init:
            sigma.detach()
            while keyboard.detach or keyboard.init:
                if keyboard.init:
                    keyboard.init = False
                    keyboard.detach = True
                    init(robot, sigma)
                time.sleep(0.1)
            sigma.resume()
            last_time = None

        # color_image2, depth_image2 = camera2.get_data()
        tcpPose, jointPose, _, _ = robot.get_robot_state()
        # print(tcpPose, jointPose)
        # t = time.time()
        diff_p, diff_r, width = sigma.get_control()  # ~0.01s for rpc
        # print(f'{time.time()-t}s')
        # print(width)
        diff_p = diff_p + robot.init_pose[:3]
        diff_p = np.clip(diff_p, [0.1, -0.5, 0.135], [0.9, 0.5, 0.5])  # for safty
        # print(diff_p)
        diff_r = diff_r * R.from_quat(robot.init_pose[3:], scalar_first=True)
        # Send command.
        robot.send_tcp_pose(
            np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0)
        )
        gripper.move(width)
        cv2.imshow('side', cam_data[0][0])
        cv2.waitKey(1)
        if not keyboard.start:
            continue
        now = time.time()
        if (
            last_time is not None
            and (now - start_time) // 0.1 == (last_time - start_time) // 0.1
        ):
            continue
        cnt += 1
        if last_time is not None:
            total_time.append(now - last_time)
        last_time = now

        start_time = time.time() if start_time is None else start_time
        for (color_image, depth_image), color_path, depth_path in zip(
            cam_data, color_dir, depth_dir
        ):
            queue.put(0)
            if queue.qsize() > 20:
                print(f'Save image queue size is too large: {queue.qsize()}')
            pool.apply_async(
                save_data,
                args=(
                    color_image.copy(),
                    depth_image.copy(),
                    os.path.join(color_path, f'{curr_time}.png'),
                    os.path.join(depth_path, f'{curr_time}.png'),
                ),
            )

        # Image.fromarray(color_image).save(os.path.join(color_dir, f'{curr_time}.png'))
        # Image.fromarray(depth_image).save(os.path.join(depth_dir, f'{curr_time}.png'))
        # TODO: 保存的是command还是state
        # TCP 只取了[:7]
        np.save(os.path.join(tcp_dir, f'{curr_time}.npy'), tcpPose)
        # joint 没有用上
        np.save(os.path.join(joint_dir, f'{curr_time}.npy'), jointPose)
        # gripper 只取了[0]
        np.save(os.path.join(gripper_dir, f'{curr_time}.npy'), [width])
        np.save(
            os.path.join(action_dir, f'{curr_time}.npy'),
            np.concatenate((diff_p, diff_r.as_quat(scalar_first=True))),
        )
    if not keyboard.start or keyboard.quit or keyboard.discard:
        print('WARNING: discard the demo!')
        time.sleep(5)
        shutil.rmtree(demo_path)
        return
    global CNT
    CNT += 1
    print(f'{CNT} saved:', demo_path, 'fps:', len(total_time) / sum(total_time))
    meta = {'finish_time': int(time.time() * 1000)}
    with open(os.path.join(demo_path, 'metadata.json'), 'w') as f:
        json.dump(meta, f)


def main():
    robot = FlexivRobot('192.168.2.100')

    gripper = FlexivGripper(robot)
    camera = [CameraD400(s) for s in camera_serial]
    sigma = Sigma7(pos_scale=4)
    # sigma = Sigma7RPC()
    keyboard = Keyboard()
    # robot, gripper, sigma = None, None, None
    # try:
    pool = multiprocessing.Pool(8)
    while not keyboard.quit:
        # init_image(camera[0])
        # global reference
        # if reference is None:
        #     color, depth = camera[0].get_data()
        #     reference = color
        record(robot, gripper, camera, sigma, keyboard, pool)
    # except Exception as e:
    #     print('ERROR:', e)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
