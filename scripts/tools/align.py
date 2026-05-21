import cv2
import numpy as np
from PIL import Image
import tyro

from rbot.device.camera import CameraD400


def main(temp: str, cam: str = '750612070265'):
    img = np.array(Image.open(temp))

    cam = CameraD400(cam)

    while True:
        color, depth = cam.get_data()
        vis = color * 0.5 + img * 0.5
        vis = vis / 255.0
        cv2.imshow('a', vis)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    tyro.cli(main)
