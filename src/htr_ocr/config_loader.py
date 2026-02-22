from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def project_root() -> Path:
    # .../src/htr_ocr/config_loader.py == root
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return project_root() / "configs"


def load_cfg(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(configs_dir())):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def cfg_to_flat_dict(cfg: DictConfig) -> dict[str, Any]:
    from omegaconf import OmegaConf

    obj = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(obj, dict)
    return obj
