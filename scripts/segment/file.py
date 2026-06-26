from pathlib import Path

import numpy as np
from PIL import Image
import tyro

from rbot.segment.sam3adapter import SAM3Adapter


def main(path: Path | str):
    segmenter = SAM3Adapter()
    # path = Path('/ssd1/mzc/workspace/sam-3d-objects/notebook/00000.png')
    image = Image.open(path)
    maskes, bboxs = segmenter.segment(image)
    for i, mask in enumerate(maskes):
        mask = Image.fromarray(mask[0].cpu().numpy().astype(np.uint8) * 255)
        mask.save(path.parent / f'{i}.png')


if __name__ == '__main__':
    tyro.cli(main)
