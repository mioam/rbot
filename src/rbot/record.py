from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from PIL import Image


def save_img(
    img: np.ndarray,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


class RawDataset:
    def __init__(self, datapath: Path, exist_ok=False):
        self.datapath = datapath
        self.datapath.mkdir(parents=True, exist_ok=exist_ok)
        self.demo_path = None
        self.executor = ThreadPoolExecutor(8)

        # self.pool = multiprocessing.pool(8)
        # self.pending_count = multiprocessing.Value('i', 0)

    def new_demo(self):
        assert self.demo_path is None
        start_time = int(time.time() * 1000)
        self.demo_path = self.datapath / f'{start_time}'
        self.frames = []
        self.demo_path.mkdir(exist_ok=False)

    def add_frame(self, frame: dict):
        curr_idx = len(self.frames)
        frame = frame.copy()

        assert self.demo_path is not None
        keys = list(frame.keys())
        for key in keys:
            if 'images' in key or 'depth' in key:
                data = frame.pop(key)
                img_path = self.demo_path / f'{key}' / f'{curr_idx:05d}.png'
                # print(f'queue size too large: {self.queue.qsize()}')
                self.executor.submit(save_img, data, img_path)
                # 添加任务data, img_path
        self.frames.append(frame)

    def _join(self):
        # 等待所有任务结束
        self.executor.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(8)

    def discard(self):
        self._join()
        if self.demo_path and self.demo_path.exists():
            shutil.rmtree(self.demo_path)
        self.demo_path = None
        self.frames = []

    def save(self):
        assert self.demo_path is not None
        self._join()
        with (self.demo_path / 'data.pkl').open('wb') as f:
            pickle.dump(self.frames, f)
        print(f'Demo saved to: {self.demo_path}')
        self.demo_path = None
        self.frames = []
