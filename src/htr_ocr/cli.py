from pathlib import Path

import fire
import pandas as pd
from rich.console import Console

from htr_ocr.config_loader import load_cfg, project_root
from htr_ocr.data.iam import build_manifest
from htr_ocr.data.splits import make_group_split
from htr_ocr.utils.io import ensure_dir
from htr_ocr.utils.mlflow_utils import mlflow_run

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from htr_ocr.data.collate import collate_line_batch
from htr_ocr.data.dataset import IamLineDataset
from htr_ocr.data.samplers import BucketBatchSampler
from htr_ocr.data.transforms import make_image_transform

console = Console()


class HTRCLI:
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
                f"Manifest not found: {manifest_path}."
            )

        ensure_dir(processed_dir)

        with mlflow_run("make_splits", cfg):
            df = pd.read_parquet(manifest_path)

            strategy = str(cfg.split.strategy)
            if strategy == "writer":
                group_col = "writer_id"
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

    def inspect_data(self, *overrides: str) -> None:
        cfg = load_cfg("inspect_data", overrides=list(overrides))

        processed_dir = Path(cfg.data.processed_dir)
        split_name = str(cfg.loader.split)
        csv_path = processed_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        transform = make_image_transform(
            height=int(cfg.preprocess.height),
            keep_aspect=bool(cfg.preprocess.keep_aspect),
            tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
            tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
            tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
            to_float_tensor=True,
        )

        ds = IamLineDataset(csv_path=csv_path, transform=transform, target_height=int(cfg.preprocess.height))

        bucket_enabled = bool(cfg.loader.bucket.enabled)
        batch_size = int(cfg.loader.batch_size)

        if bucket_enabled:
            lengths = [ds.approx_resized_width(i) for i in range(len(ds))]
            if any(v is None for v in lengths):
                console.print("No requested columns")
                bucket_enabled = False
            else:
                lengths_i = [int(v) for v in lengths if v is not None]
                sampler = BucketBatchSampler(
                    lengths=lengths_i,
                    batch_size=batch_size,
                    shuffle_batches=bool(cfg.loader.shuffle),
                    seed=int(cfg.loader.bucket.seed),
                    drop_last=bool(cfg.loader.bucket.drop_last),
                )
                dl = DataLoader(
                    ds,
                    batch_sampler=sampler,
                    num_workers=int(cfg.loader.num_workers),
                    pin_memory=bool(cfg.loader.pin_memory),
                    collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
                )

        if not bucket_enabled:
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=bool(cfg.loader.shuffle),
                num_workers=int(cfg.loader.num_workers),
                pin_memory=bool(cfg.loader.pin_memory),
                collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
            )

        with mlflow_run("inspect_data", cfg, extra_tags={"split": split_name}):
            n_batches = int(cfg.loader.n_batches)
            console.print(
                f"Inspecting split={split_name} batches={n_batches} bs={batch_size} "
                f"height={int(cfg.preprocess.height)} bucket={bucket_enabled}"
            )

            for bi, batch in enumerate(dl):
                if bi >= n_batches:
                    break
                pv = batch["pixel_values"]
                widths = batch["widths"]
                w_max = int(pv.shape[-1])

                grid = make_grid(pv.detach().cpu(), nrow=2, padding=2)  # [1,Hgrid,Wgrid]
                ensure_dir(cfg.loader.samples_path)
                to_pil_image(grid).save(Path(cfg.loader.samples_path) / f"grid_{bi}.png")

                console.print(
                    f"batch[{bi}]: pixel_values={tuple(pv.shape)} w_max={w_max} "
                    f"min_w={min(widths)} max_w={max(widths)}"
                )

                t0 = batch["texts"][0]
                console.print(f"  sample text[0]: {t0[:120]}")


def main() -> None:
    fire.Fire(HTRCLI)
