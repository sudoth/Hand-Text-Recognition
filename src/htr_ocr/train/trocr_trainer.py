from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)

from htr_ocr.data.trocr_dataset import TrOCRLineDataset, build_trocr_collate
from htr_ocr.data.transforms import make_image_transform
from htr_ocr.utils.io import ensure_dir
from htr_ocr.utils.metrics import cer, wer
from htr_ocr.utils.repro import seed_everything


@dataclass
class TrainResult:
    best_checkpoint: Path
    best_val_cer: float
    best_val_wer: float


def _build_transform(cfg, is_train: bool):
    return make_image_transform(
        height=int(cfg.preprocess.height),
        keep_aspect=bool(cfg.preprocess.keep_aspect),
        tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
        tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
        tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
        augment_cfg=(cfg.augment if is_train and bool(cfg.augment.enabled) else None),
        is_train=is_train,
        fill=int(cfg.preprocess.pad_value),
        to_float_tensor=False,
    )


def _set_encoder_trainable(model: VisionEncoderDecoderModel, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def _build_scheduler(
    cfg,
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    total_train_steps: int,
    warmup_steps: int,
) -> tuple[object | None, str]:
    scheduler_cfg = getattr(cfg.train, "scheduler", None)

    # New format (analogous to vt-ctc):
    # train.scheduler.enabled/name/t_max/eta_min
    if scheduler_cfg is not None and hasattr(scheduler_cfg, "name"):
        if not bool(getattr(scheduler_cfg, "enabled", True)):
            return None, "none"

        scheduler_name = str(getattr(scheduler_cfg, "name", "cosine")).lower()
        if scheduler_name == "cosine":
            t_max = max(1, int(getattr(scheduler_cfg, "t_max", max_epochs)))
            eta_min = float(getattr(scheduler_cfg, "eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
            return scheduler, "epoch"

        raise ValueError(f"Unknown train.scheduler.name={scheduler_name}")

    # Legacy format:
    # train.scheduler: linear
    scheduler_name = str(scheduler_cfg) if scheduler_cfg is not None else "linear"
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    return scheduler, "step"


def make_dataloader(cfg, split: str, processor: TrOCRProcessor) -> DataLoader:
    processed_dir = Path(cfg.data.processed_dir)
    csv_path = processed_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    is_train = split == "train"
    transform = _build_transform(cfg, is_train=is_train)

    ds = TrOCRLineDataset(
        csv_path=csv_path,
        transform=transform,
    )

    dl = DataLoader(
        ds,
        batch_size=int(cfg.loader.batch_size),
        shuffle=bool(cfg.loader.shuffle) if is_train else False,
        num_workers=int(cfg.loader.num_workers),
        pin_memory=bool(cfg.loader.pin_memory),
        collate_fn=build_trocr_collate(
            processor=processor,
            max_target_length=int(cfg.model.max_target_length),
        ),
    )
    return dl


@torch.inference_mode()
def evaluate(model, processor, dl, device: torch.device, generate_cfg) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    n = 0

    for batch in tqdm(dl, desc="eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        texts = batch["texts"]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        generated_ids = model.generate(
            pixel_values,
            num_beams=int(generate_cfg.num_beams),
            max_new_tokens=int(generate_cfg.max_new_tokens),
            length_penalty=float(generate_cfg.length_penalty),
            early_stopping=bool(generate_cfg.early_stopping),
            no_repeat_ngram_size=int(generate_cfg.no_repeat_ngram_size),
        )
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        bs = len(texts)
        total_loss += float(loss.item()) * bs
        total_cer += sum(cer(pred, gt) for pred, gt in zip(preds, texts))
        total_wer += sum(wer(pred, gt) for pred, gt in zip(preds, texts))
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "cer": total_cer / max(1, n),
        "wer": total_wer / max(1, n),
    }


def train_trocr(cfg) -> TrainResult:
    seed_everything(int(cfg.train.seed), deterministic=bool(getattr(cfg.train, "deterministic", True)))
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained(str(cfg.model.pretrained_name))
    model = VisionEncoderDecoderModel.from_pretrained(
        str(cfg.model.pretrained_name),
        use_safetensors=True,
    ).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    train_dl = make_dataloader(cfg, "train", processor)
    val_dl = make_dataloader(cfg, "val", processor)

    freeze_epochs = int(cfg.model.freeze_encoder_epochs)
    if freeze_epochs > 0:
        _set_encoder_trainable(model, trainable=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.betas[0]), float(cfg.train.betas[1])),
        eps=float(cfg.train.eps),
    )

    steps_per_epoch = max(1, len(train_dl))
    grad_accum_steps = int(cfg.train.grad_accum_steps)
    total_train_steps = int(cfg.train.max_epochs) * max(1, (steps_per_epoch + grad_accum_steps - 1) // grad_accum_steps)
    warmup_steps = int(float(cfg.train.warmup_ratio) * total_train_steps)

    scheduler, scheduler_step_mode = _build_scheduler(
        cfg,
        optimizer,
        max_epochs=int(cfg.train.max_epochs),
        total_train_steps=total_train_steps,
        warmup_steps=warmup_steps,
    )

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = Path(cfg.train.runs_dir) / "trocr"
    best_dir = run_dir / "best"
    last_dir = run_dir / "last"
    ensure_dir(best_dir)
    ensure_dir(last_dir)

    best_val_cer = float("inf")
    best_val_wer = float("inf")
    bad_epochs = 0

    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _set_encoder_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg.train.lr),
                weight_decay=float(cfg.train.weight_decay),
                betas=(float(cfg.train.betas[0]), float(cfg.train.betas[1])),
                eps=float(cfg.train.eps),
            )
            scheduler, scheduler_step_mode = _build_scheduler(
                cfg,
                optimizer,
                max_epochs=int(cfg.train.max_epochs),
                total_train_steps=total_train_steps,
                warmup_steps=warmup_steps,
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        seen = 0

        pbar = tqdm(train_dl, desc=f"train e{epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            bs = int(pixel_values.shape[0])

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            if step % grad_accum_steps == 0 or step == len(train_dl):
                if float(cfg.train.max_grad_norm) > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_mode == "step":
                    scheduler.step()

            epoch_loss += float(loss.item()) * bs
            seen += bs
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = epoch_loss / max(1, seen)
        val_metrics = evaluate(model, processor, val_dl, device, generate_cfg=cfg.generate)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val_cer", val_metrics["cer"], step=epoch)
        mlflow.log_metric("val_wer", val_metrics["wer"], step=epoch)
        mlflow.log_metric("lr", float(optimizer.param_groups[0]["lr"]), step=epoch)

        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step()

        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        improved = val_metrics["cer"] < best_val_cer
        if improved:
            best_val_cer = val_metrics["cer"]
            best_val_wer = val_metrics["wer"]
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            if bool(getattr(cfg.train, "log_checkpoint_to_mlflow", True)):
                mlflow.log_artifacts(str(best_dir), artifact_path="checkpoints/best")
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg.train.patience):
            break

    return TrainResult(
        best_checkpoint=best_dir,
        best_val_cer=best_val_cer,
        best_val_wer=best_val_wer,
    )
