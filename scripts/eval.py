from pathlib import Path
import time

import numpy as np
from PIL import Image
import tyro

from rbot.agent import Agent
from rbot.common.image_util import center_crop_resize
from rbot.common.precise_sleep import precise_wait
from rbot.record import RawDataset
from rbot.utils.tools import imshow

SIZE = [256, 256]
FPS = 10


def main(ckpt: str = 'checkpoints/pi0_my/my_experiment/29999', remote=False, port=8000):
    agent = Agent(camera_serials=['750612070265', '244222073667'])
    init_pose = (0.5, 0, 0.4, 0, np.sin(0), np.cos(0), 0)
    agent.set_tcp_pose(init_pose)
    time.sleep(3)

    if not remote:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as _config

        config_name = Path(ckpt).parts[1]
        config = _config.get_config(config_name)
        checkpoint_dir = download.maybe_download(ckpt)
        # checkpoint_dir = download.maybe_download('checkpoints/pi0_my/pi0_carrot/29999')
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
    else:
        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(
            host='localhost', port=port
        )
    dataset = RawDataset(Path('/ssd1/mzc/data/replay'), exist_ok=True)
    dataset.new_demo()
    try:
        while True:
            frame = agent.get_frame()

            # show(frame)
            for key in frame:
                if 'images' in key:
                    img = frame[key]
                    resized_img = center_crop_resize(Image.fromarray(img), SIZE)
                    frame[key] = np.array(resized_img, dtype=np.uint8)

                    imshow(key, frame[key])
            frame['prompt'] = ''
            data = policy.infer(frame)
            action_chunk = data['actions']

            state = True
            for i in range(10):
                frame_start = time.monotonic()
                frame1 = agent.get_frame()
                print(action_chunk[i])
                agent.set_tcp_pose(action_chunk[i, :-1])
                agent.set_gripper_width(action_chunk[i, -1])
                dataset.add_frame(frame1)

                precise_wait(frame_start + 1.0 / FPS)
                # if state != (action_chunk[i, -1] > 500):
                #     state = not state
                #     agent.set_gripper_width(1000 if state else 0)
    except KeyboardInterrupt:
        print('KeyboardInterrupt!')
    finally:
        dataset.save()


if __name__ == '__main__':
    tyro.cli(main)
