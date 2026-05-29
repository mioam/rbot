# 这个文件用于简化对照实验的config设置。从原始的config中抽离关键参数
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf
from openpi.training.config import TrainConfig

@dataclass
class _ModelConfig:
    name: str

@dataclass
class _TrainConfig:
    batch_size: int = 80
    num_steps: int = 10_000
    repo_id: str = ''

@dataclass
class _DataConfig:
    state:str
    action:str
    delta:bool

@dataclass
class _InferConfig:
    ckpt_dir: str = ''

@dataclass
class Config:
    exp_name: str
    model: _ModelConfig
    train: _TrainConfig
    data: _DataConfig
    infer: _InferConfig



def save_config(cfg: Config, path: str | Path):
    OmegaConf.save(
        OmegaConf.structured(cfg),
        path,
    )


def load_config(path: str | Path):
    schema = OmegaConf.structured(Config)
    loaded = OmegaConf.load(path)

    merged = OmegaConf.merge(schema, loaded)

    return cast(Config, OmegaConf.to_object(merged))
