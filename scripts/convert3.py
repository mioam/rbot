from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image
from termcolor import cprint
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rbot.common.image_util import Cropper


class RawDatasetReader:
    """
    读取 RawDataset 保存的数据。
    """

    def __init__(self, datapath: str | Path):
        self.datapath = Path(datapath)
        self.demo_list = sorted(
            int(p.name)
            for p in self.datapath.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        if not self.demo_list:
            raise ValueError(f'No numeric demo directories found in {self.datapath}')
        self.target_size = [256, 256]
        self.cropper = Cropper()

    def get_image(self, img_path: Path):
        img = Image.open(img_path)
        img = np.array(img)
        return img

    def get_feature(self):
        demo_dir = Path(self.datapath / f'{self.demo_list[0]}')
        self.cam_features = {}
        cam_data = {}
        for p in demo_dir.iterdir():
            if p.is_dir():
                if 'depth' in p.name:
                    continue
                cam_data[p.name] = self.get_image(p / f'{0:05}.png')
        self.cropper.crop_frame(cam_data)
        for key in cam_data:
            img = cam_data[key]
            print(f'{key}: {img.shape}, {img.dtype}')
            self.cam_features[key] = {
                'dtype': 'image',
                'shape': img.shape,
                'names': ['height', 'width', 'channel'],
            }

        self.features = self.cam_features | {
            'observation.state.joint': {
                'dtype': 'float32',
                'shape': (7,),
                'names': ['joint'],
            },
            'observation.state.tcp': {
                'dtype': 'float32',
                'shape': (7,),
                'names': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'],
            },
            'observation.state.gripper': {
                'dtype': 'float32',
                'shape': (1,),
                'names': ['gripper'],
            },
            'actions': {
                'dtype': 'float32',
                'shape': (8,),
                'names': ['actions'],
            },
        }
        return self.features

    def get_demo_dirs(self):
        return [self.datapath / f'{name}' for name in self.demo_list]

    def iter_frames(self, demo_dir: str | Path):
        """
        遍历某个 demo 的所有 frame。

        返回的 frame 与 add_frame 输入格式一致：
        - 普通字段直接来自 data.pkl
        - images/depth 自动读取 png 并恢复为 np.ndarray
        """
        demo_dir = Path(demo_dir)

        data_path = demo_dir / 'data.pkl'

        with data_path.open('rb') as f:
            frames = pickle.load(f)

        for idx, frame in enumerate(frames):
            frame = frame.copy()
            if 'action' in frame:
                frame['actions'] = frame.pop('action')
            frame = {k: frame[k] for k in frame if k in self.features or k == 'task'}

            missing = set(self.features) - set(self.cam_features) - set(frame)
            if missing:
                raise ValueError(
                    f'{demo_dir}/data.pkl frame {idx} is missing features: {sorted(missing)}'
                )

            # 自动寻找对应图片目录
            for key in self.cam_features:
                img_path = demo_dir / key / f'{idx:05d}.png'
                img = self.get_image(img_path)
                frame[key] = img
            self.cropper.crop_frame(frame)

            yield frame


def main(
    path: str,
    name: str,
    task: str,
    fps: int = 10,
):
    """Convert recorded raw demos to a finalized LeRobot Dataset v3 dataset."""

    reader = RawDatasetReader(path)
    features = reader.get_feature()

    REPO_NAME = f'miaom/{name}'

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        fps=fps,
        features=features,
    )
    try:
        for demo_dir in reader.get_demo_dirs():
            cprint(demo_dir, 'green')
            frame_count = 0
            for frame in reader.iter_frames(demo_dir):
                # Dataset v3 requires every frame to carry its task prompt. `record3.py`
                # stores an empty task when no prompt was given during recording, so
                # treat that as absent and use the explicit CLI default.
                if not frame.get('task'):
                    frame['task'] = task
                dataset.add_frame(frame)
                frame_count += 1
            if frame_count == 0:
                raise ValueError(f'{demo_dir}/data.pkl contains no frames')
            dataset.save_episode()
    finally:
        # v3 parquet metadata and pending video encoders are not valid until this
        # is called. Keep it explicit rather than relying on object finalization.
        dataset.finalize()
    with (dataset.root / 'log').open('w') as f:
        f.write(f'{sys.argv}')


if __name__ == '__main__':
    tyro.cli(main)
