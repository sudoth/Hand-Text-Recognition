from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd
from rich.console import Console

from htr_ocr.config_loader import load_cfg
from htr_ocr.data.iam import build_manifest
from htr_ocr.data.splits import make_group_split
from htr_ocr.utils.io import ensure_dir
from htr_ocr.utils.mlflow_utils import mlflow_run

console = Console()


class HTRCLI:
    """
    Все параметры задаём через Hydra overrides:
    uv run htr make_manifest data.raw_dir=data/raw preprocess.height=128
    """

    def make_manifest(self, *overrides: str) -> None:
        cfg = load_cfg("make_manifest", overrides=list(overrides))

        ensure_dir(cfg.data.processed_dir)

        with mlflow_run("make_manifest", cfg):
            df = build_manifest(
                images_root=cfg.data.images_root,
                annotations_path=cfg.data.annotations_path,
                forms_path=getattr(cfg.data, 'forms_path', None),
                keep_status=list(cfg.data.keep_status),
                limit=int(cfg.data.limit),
            )

            manifest_path = Path(cfg.data.manifest_path)
            df.to_parquet(manifest_path, index=False)
            console.print(f"Saved manifest: {manifest_path} (rows={len(df)})")

    def make_splits(self, *overrides: str) -> None:
        cfg = load_cfg("make_splits", overrides=list(overrides))

        processed_dir = Path(cfg.data.processed_dir)
        manifest_path = Path(cfg.data.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run `htr make_manifest` first."
            )

        ensure_dir(processed_dir)

        with mlflow_run("make_splits", cfg):
            df = pd.read_parquet(manifest_path)

            strategy = str(cfg.split.strategy)
            if strategy == "writer":
                group_col = "writer_id"
                if df[group_col].isna().any():
                    missing = int(df[group_col].isna().sum())
                    raise ValueError(
                        f"writer_id missing for {missing} rows. "
                        "Rebuild manifest with data.forms_path pointing to ascii/forms.txt."
                    )
            elif strategy == "form":
                group_col = "form_id"
            else:
                raise ValueError(f"Unknown split.strategy={strategy} (expected 'form' or 'writer')")

            train_df, val_df, test_df = make_group_split(
                df,
                group_col=group_col,
                seed=int(cfg.split.seed),
                train=float(cfg.split.train),
                val=float(cfg.split.val),
                test=float(cfg.split.test),
            )

            (processed_dir / "train.csv").write_text(train_df.to_csv(index=False), encoding="utf-8")
            (processed_dir / "val.csv").write_text(val_df.to_csv(index=False), encoding="utf-8")
            (processed_dir / "test.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")

            console.print(
                "Saved splits to data/processed: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} (group={group_col})"
            )


def main() -> None:
    fire.Fire(HTRCLI)
