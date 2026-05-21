#################################### For Image ####################################

from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
import torch


class SAM3Adapter:
    def __init__(self):
        # Load the model
        model = build_sam3_image_model(
            checkpoint_path='/ssd1/mzc/workspace/sam3/ckpts/sam3.pt'
        )
        self.processor = Sam3Processor(model)

    def segment(self, image: Image, prompt='objects'):

        # Load an image
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            # Get the masks, bounding boxes, and scores
            masks, boxes, scores = output['masks'], output['boxes'], output['scores']
            return masks, boxes
