from __future__ import annotations

import contextlib
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlflow

from htr_ocr.config_loader import cfg_to_flat_dict, project_root


@dataclass
class MlflowCfg:
    enabled: bool
    tracking_uri: str
    experiment: str
    tags: dict[str, str]


def _get_git_commit() -> str:
    # Без зависимости от gitpython
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root())
        return out.decode().strip()
    except Exception:
        return "unknown"


def setup_mlflow(cfg: dict) -> None:
    if not cfg.get("enabled", True):
        return

    tracking_uri = cfg.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(cfg.get("experiment", "htr-diploma"))


@contextlib.contextmanager
def mlflow_run(run_name: str, cfg, extra_tags: dict[str, str] | None = None) -> Iterator[None]:
    mlflow_cfg = cfg.get("mlflow", {})
    enabled = bool(mlflow_cfg.get("enabled", True))

    if not enabled:
        yield
        return

    setup_mlflow(mlflow_cfg)

    tags = dict(mlflow_cfg.get("tags", {}) or {})
    if extra_tags:
        tags.update(extra_tags)

    tags.setdefault("git_commit", _get_git_commit())
    tags.setdefault("python", sys.version.split()[0])
    tags.setdefault("platform", platform.platform())

    with mlflow.start_run(run_name=run_name, tags=tags):
        # логируем всю конфигурацию как параметры (в плоском виде)
        flat = cfg_to_flat_dict(cfg)
        mlflow.log_params(_flatten_for_mlflow(flat))

        # сохраняем конфиг как artifact
        tmp = Path(".cache_mlflow")
        tmp.mkdir(exist_ok=True)
        cfg_path = tmp / "config_resolved.yaml"
        from omegaconf import OmegaConf

        OmegaConf.save(config=cfg, f=str(cfg_path))
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        yield


def _flatten_for_mlflow(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_mlflow(v, prefix=key))
        else:
            out[key] = str(v)
    return out
