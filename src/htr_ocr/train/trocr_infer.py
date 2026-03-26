from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from htr_ocr.data.transforms import make_image_transform
from htr_ocr.train.trocr_common import fix_trocr_sinusoidal_positional_weights


def load_checkpoint(path: Path, device: torch.device) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    processor = TrOCRProcessor.from_pretrained(str(path))
    model = VisionEncoderDecoderModel.from_pretrained(
        str(path),
        use_safetensors=True,
    ).to(device)
    fix_trocr_sinusoidal_positional_weights(model, device)
    model.eval()
    return model, processor


@torch.inference_mode()
def infer_one(
    checkpoint_path: Path,
    image_path: Path,
    height: int,
    keep_aspect: bool,
    pad_value: int,
    device_str: str,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    no_repeat_ngram_size: int = 0,
) -> str:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, processor = load_checkpoint(checkpoint_path, device)

    transform = make_image_transform(
        height=int(height),
        keep_aspect=bool(keep_aspect),
        tight_crop_enabled=False,
        tight_crop_threshold=0,
        tight_crop_margin=0,
        augment_cfg=None,
        is_train=False,
        fill=int(pad_value),
        to_float_tensor=False,
    )

    image = Image.open(image_path).convert("L")
    image = transform(image)
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        num_beams=int(num_beams),
        max_new_tokens=int(max_new_tokens),
        length_penalty=float(length_penalty),
        early_stopping=bool(early_stopping),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
    )
    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return pred
