from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image
from termcolor import cprint
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rbot.common.image_util import center_crop_resize


class RawDatasetReader:
    """
    读取 RawDataset 保存的数据。
    """

    def __init__(self, datapath: str):
        self.datapath = Path(datapath)
        self.demo_list = sorted([
            int(p.name) for p in self.datapath.iterdir() if p.is_dir()
        ])
        self.target_size = [256, 256]

    def get_image(self, img_path: Path):

        img = Image.open(img_path)
        img = center_crop_resize(img, self.target_size)
        img = np.array(img)
        return img

    def get_feature(self):
        demo_dir = Path(self.datapath / f'{self.demo_list[0]}')
        self.cam_features = {}
        for p in demo_dir.iterdir():
            if p.is_dir():
                if 'depth' in p.name:
                    continue
                img = self.get_image(p / f'{0:05}.png')
                print(f'{p.name}: {img.shape}, {img.dtype}')
                self.cam_features[p.name] = {
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

            # 自动寻找对应图片目录
            for key in self.cam_features:
                img_path = demo_dir / key / f'{idx:05d}.png'
                img = self.get_image(img_path)
                frame[key] = img

            yield frame


def main(path: str, name: str, FPS=10):
    reader = RawDatasetReader(path)
    features = reader.get_feature()

    REPO_NAME = f'miaom/{name}'

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        fps=FPS,
        features=features,
        # streaming_encoding=True,
    )
    for demo_dir in reader.get_demo_dirs():
        cprint(demo_dir, 'green')
        for frame in reader.iter_frames(demo_dir):
            dataset.add_frame(frame)
        dataset.save_episode()
    with (dataset.root / 'log').open('w') as f:
        f.write(f'{sys.argv}')
    # dataset.finalize()


if __name__ == '__main__':
    tyro.cli(main)
