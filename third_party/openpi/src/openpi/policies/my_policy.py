import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi import transforms
from openpi.models import model as _model


def action_encoder(data, action_config):
    action = []
    for name in action_config:
        if name == "tcp":
            action.append(data["actions"])
        elif name == "gripper":
            gripper_pos = np.asarray(data["observation/state-gripper"])
            if gripper_pos.ndim == 0:
                gripper_pos = gripper_pos[np.newaxis]
            action.append(gripper_pos)
        else:
            raise NotImplementedError

    return np.concatenate(action)


def state_parser(data, state_config):
    state = []
    for name in state_config:
        if name == "tcp":
            state.append(data["observation/state-tcp"])
        elif name == "gripper":
            gripper_pos = np.asarray(data["observation/state-gripper"])
            if gripper_pos.ndim == 0:
                gripper_pos = gripper_pos[np.newaxis]
            state.append(gripper_pos)
        else:
            raise NotImplementedError(state)

    return np.concatenate(state)


def make_my_example() -> dict:
    """Creates a random input example for the My policy."""
    return {
        "observation/state-tcp": np.random.rand(7),
        "observation/state-gripper": np.random.rand(1),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MyInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    state_config: list[str] = dataclasses.field(default_factory=["joint"])
    action_config: str = "tcp_quat"

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        # print(data.keys())

        base_image = _parse_image(data["observation.images.750612070265"])
        wrist_image = _parse_image(data["observation.images.244222073667"])

        # gripper_pos = np.asarray(data["observation/state-gripper"])
        # if gripper_pos.ndim == 0:
        #     gripper_pos = gripper_pos[np.newaxis]
        # state = np.concatenate([data["observation/state-tcp"], gripper_pos])
        tcp = data["observation.state.tcp"]
        gripper = data["observation.state.gripper"]
        joint = data["observation.state.joint"]
        if gripper.ndim == 0:
            gripper = gripper[None]
        state = []
        for key in self.state_config:
            if key == "joint":
                state.append(joint)
            elif key == "gripper":
                state.append(gripper)
            elif key == "tcp_quat":
                state.append(tcp)
            elif key == "tcp_eular":
                tcp_eular = tcp.copy()
                tcp_eular[3:] = R.from_quat(tcp[3:], scalar_first=True).as_eular()
                state.append(tcp_eular)

        state = np.concatenate(state, 0)

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            if self.action_config == "tcp_quat":
                inputs["actions"] = data["actions"]
            elif self.action_config == "tcp_eular":
                inputs["actions"] = np.concatenate([joint, gripper], 0)
                inputs["actions"] = data["actisons"]
            elif self.action_config == "joint":
                inputs["actions"] = np.concatenate([joint, gripper], 0)

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MyOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_config: list[str]

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :8])}
