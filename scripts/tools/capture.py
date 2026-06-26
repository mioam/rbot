from pathlib import Path

import cv2
from PIL import Image
import tyro

from rbot.device.camera import CameraD400


def main(cam: str = '750612070265', out: Path | str = './outputs/scene'):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    cam = CameraD400(cam)

    while True:
        color, depth = cam.get_data()
        cv2.imshow('a', color)
        if cv2.waitKey(1) == ord('q'):
            break

    Image.fromarray(color[..., ::-1]).save(out / 'color.png')
    Image.fromarray(depth).save(out / 'depth.png')


if __name__ == '__main__':
    tyro.cli(main)
