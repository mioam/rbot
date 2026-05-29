import dataclasses
import pathlib

import einops
import numpy as np
from typing_extensions import override
import tyro

from openpi import transforms
from openpi.models import pi0_config
import openpi.models.model as _model
from openpi.training import weight_loaders
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
)
import openpi.transforms as _transforms
from rbot.common.transformation import mat_to_xyz_rot, xyz_rot_to_mat, xyz_rot_transform
from rbot.openpi.config_builder import Config

STATE_MAPPING = {
    'q': 'quaterion (scalar first)',
    'e': 'eular XYZ',
    'r': 'rot6d',
    'j': 'joint',
    'g': 'gripper',
}

ACTION_MAPPING = {
    'q': 'quaterion (scalar first)',
    'e': 'eular XYZ',
    'r': 'rot6d',
    'j': 'next joint',  #  ?
}


def make_my_example() -> dict:
    """Creates a random input example for the My policy."""
    return {
        'observation/state-tcp': np.random.rand(7),
        'observation/state-gripper': np.random.rand(1),
        'observation/image': np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        'prompt': 'do something',
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, 'c h w -> h w c')
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
    state_config: str
    action_config: str
    delta: bool

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
        # from rbot.utils.tools import show

        # show(data)
        base_image = _parse_image(data['observation.images.750612070265'])
        wrist_image = _parse_image(data['observation.images.244222073667'])

        tcp = data['observation.state.tcp']
        gripper = data['observation.state.gripper']
        joint = data['observation.state.joint']
        if gripper.ndim == 0:
            gripper = gripper[None]
        state = []
        for key in self.state_config:
            if key == 'j':
                state.append(joint)
            elif key == 'g':
                state.append(gripper)
            elif key == 'q':
                state.append(tcp)
            elif key == 'e':
                tcp_eular = xyz_rot_transform(
                    tcp,
                    from_rep='quaternion',
                    to_rep='euler_angles',
                    to_convention='XYZ',
                )
                state.append(tcp_eular)
            elif key == 'r':
                tcp_rot6d = xyz_rot_transform(
                    tcp,
                    from_rep='quaternion',
                    to_rep='rotation_6d',
                )
                state.append(tcp_rot6d)

        state = np.concatenate(state, 0)
        state_matrix = xyz_rot_to_mat(
            data['observation.state.tcp'],
            from_rep='quaternion',
        )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            'state': state,
            'state_matrix': state_matrix,
            'image': {
                'base_0_rgb': base_image,
                'left_wrist_0_rgb': wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                'right_wrist_0_rgb': np.zeros_like(base_image),
            },
            'image_mask': {
                'base_0_rgb': np.True_,
                'left_wrist_0_rgb': np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                'right_wrist_0_rgb': np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if 'actions' in data:
            tcp_quat = data['actions'][..., :7]
            xyz = data['actions'][..., :3]
            quat = data['actions'][..., 3:7]
            gripper = data['actions'][..., 7:]
            if not self.delta:
                if self.action_config == 'q':
                    inputs['actions'] = np.concatenate([tcp_quat, gripper], -1)
                elif self.action_config == 'e':
                    tcp_eular = xyz_rot_transform(
                        tcp_quat,
                        from_rep='quaternion',
                        to_rep='euler_angles',
                        to_convention='XYZ',
                    )
                    inputs['actions'] = np.concatenate([tcp_eular, gripper], -1)
                elif self.action_config == 'r':
                    tcp_rot6d = xyz_rot_transform(
                        tcp_quat,
                        from_rep='quaternion',
                        to_rep='rotation_6d',
                    )
                    inputs['actions'] = np.concatenate([tcp_rot6d, gripper], -1)
                elif self.action_config == 'j':
                    raise NotImplementedError
            else:
                tcp_matrix = xyz_rot_to_mat(
                    tcp_quat,
                    from_rep='quaternion',
                )
                delta_tcp_matrix = np.linalg.inv(state_matrix) @ tcp_matrix

                if self.action_config == 'q':
                    tcp_quat = mat_to_xyz_rot(
                        delta_tcp_matrix,
                        to_rep='quaternion',
                    )
                    inputs['actions'] = np.concatenate([tcp_quat, gripper], -1)
                elif self.action_config == 'e':
                    tcp_eular = mat_to_xyz_rot(
                        delta_tcp_matrix,
                        to_rep='euler_angles',
                        to_convention='XYZ',
                    )
                    inputs['actions'] = np.concatenate([tcp_eular, gripper], -1)
                elif self.action_config == 'r':
                    tcp_rot6d = mat_to_xyz_rot(
                        delta_tcp_matrix,
                        to_rep='rotation_6d',
                    )
                    inputs['actions'] = np.concatenate([tcp_rot6d, gripper], -1)
                elif self.action_config == 'j':
                    raise NotImplementedError

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if 'prompt' in data:
            inputs['prompt'] = data['prompt']

        return inputs


@dataclasses.dataclass(frozen=True)
class MyOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_config: str
    delta: bool

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        # return data
        # data['actions] 是模型预测的，和Inputs中返回的actions一致。
        # 这个东西需要复原delta和旋转的表示，变成xyz quat gripper的原始action
        if self.action_config == 'q':
            actions = np.asarray(data['actions'][:, :8])
            gripper = actions[:, -1]
            tcp_matrix = xyz_rot_to_mat(
                actions[:, :7],
                from_rep='quaternion',
            )
        elif self.action_config == 'e':
            actions = np.asarray(data['actions'][:, :7])
            gripper = actions[:, -1]
            tcp_matrix = xyz_rot_to_mat(
                actions[:, :6],
                from_rep='euler_angles',
                from_convention='XYZ',
            )
        elif self.action_config == 'r':
            actions = np.asarray(data['actions'][:, :10])
            gripper = actions[:, -1]
            tcp_matrix = xyz_rot_to_mat(
                actions[:, :9],
                from_rep='rotation_6d',
            )
        elif self.action_config == 'j':
            raise NotImplementedError

        if self.delta:
            state_matrix = data['state_matrix']
            tcp_matrix = state_matrix @ tcp_matrix

        tcp_quat = mat_to_xyz_rot(
            tcp_matrix,
            to_rep='quaternion',
        )

        return {'actions': np.concatenate([tcp_quat, gripper[..., None]], -1)}


@dataclasses.dataclass(frozen=True)
class LeRobotMyDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    state: str = ''
    action: str = ''
    delta: bool = False

    @property
    def name(self):
        return f'{self.state}_{self.action}_{"rel" if self.delta else "raw"}'

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                # _transforms.RepackTransform({})
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                MyInputs(
                    model_type=model_config.model_type,
                    state_config=self.state,
                    action_config=self.action,
                    delta=self.delta,
                ),
            ],
            outputs=[MyOutputs(action_config=self.action, delta=self.delta)],
        )

        # if self.extra_delta_transform:
        #     delta_action_mask = _transforms.make_bool_mask(7)
        #     data_transforms = data_transforms.push(
        #         inputs=[_transforms.DeltaActions(delta_action_mask)],
        #         outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        #     )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # action_sequence_keys=("action",),
        )


def _build_config():
    BATCH_SIZE = 80
    STEPS = 10_000
    NUM_WORKER = 30

    defaults = {
        'PI0': TrainConfig(
            name='pi0',
            model=pi0_config.Pi0Config(),
            data=LeRobotMyDataConfig(
                repo_id=tyro.MISSING,
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
                delta=tyro.MISSING,
                state=tyro.MISSING,
                action=tyro.MISSING,
                # assets=AssetsConfig(asset_id=data_name),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                'gs://openpi-assets/checkpoints/pi0_base/params'
            ),
            num_train_steps=STEPS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKER,
        ),
        'PI05': TrainConfig(
            name='pi05',
            model=pi0_config.Pi0Config(
                pi05=True,
                # action_horizon=16,
                # discrete_state_input=False,
            ),
            data=LeRobotMyDataConfig(
                repo_id=tyro.MISSING,
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
                delta=tyro.MISSING,
                state=tyro.MISSING,
                action=tyro.MISSING,
                # assets=AssetsConfig(asset_id=data_name),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                'gs://openpi-assets/checkpoints/pi05_base/params'
            ),
            num_train_steps=STEPS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKER,
        ),
    }

    config = tyro.extras.overridable_config_cli({
        k: (k, v) for k, v in defaults.items()
    })
    assert type(config.data) is LeRobotMyDataConfig
    config = dataclasses.replace(config, name=f'{config.name}_{config.data.name}')
    return config


def build_config(config: Config):
    if config.model.name == 'pi0':
        return TrainConfig(
            exp_name=config.exp_name,
            name='pi0',
            model=pi0_config.Pi0Config(),
            data=LeRobotMyDataConfig(
                repo_id=config.train.repo_id,
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
                delta=config.data.delta,
                state=config.data.state,
                action=config.data.action,
                # assets=AssetsConfig(asset_id=data_name),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                'gs://openpi-assets/checkpoints/pi0_base/params'
            ),
            num_train_steps=config.train.num_steps,
            batch_size=config.train.batch_size,
            num_workers=30,
        )
    if config.model.name == 'pi05':
        return TrainConfig(
            exp_name=config.exp_name,
            name='pi05',
            model=pi0_config.Pi0Config(
                pi05=True,
                # action_horizon=16,
                # discrete_state_input=False,
            ),
            data=LeRobotMyDataConfig(
                repo_id=config.train.repo_id,
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
                delta=config.data.delta,
                state=config.data.state,
                action=config.data.action,
                # assets=AssetsConfig(asset_id=data_name),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                'gs://openpi-assets/checkpoints/pi05_base/params'
            ),
            num_train_steps=config.train.num_steps,
            batch_size=config.train.batch_size,
            num_workers=30,
        )
    raise ValueError()
