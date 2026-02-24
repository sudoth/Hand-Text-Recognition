from dataclasses import dataclass
from typing import Callable

import torch
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image


@dataclass(frozen=True)
class ResizeToHeight:
    height: int = 128
    keep_aspect: bool = True

    def __call__(self, img: Image.Image) -> Image.Image:
        h = int(self.height)
        if h <= 0:
            raise ValueError("height must be > 0")

        img = img.convert("L")
        w0, h0 = img.size
        if h0 <= 0:
            return img

        if self.keep_aspect:
            w = max(1, int(round(w0 * (h / h0))))
            return img.resize((w, h), resample=Image.Resampling.BILINEAR)
        return img.resize((w0, h), resample=Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class TightCrop:
    enabled: bool = False
    threshold: int = 245
    margin: int = 2

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img

        img = img.convert("L")
        bw = img.point(lambda p: 255 if p < self.threshold else 0, mode="L")
        bbox = bw.getbbox()
        if bbox is None:
            return img

        left, upper, right, lower = bbox
        m = int(self.margin)
        left = max(0, left - m)
        upper = max(0, upper - m)
        right = min(img.size[0], right + m)
        lower = min(img.size[1], lower + m)
        return img.crop((left, upper, right, lower))


def make_image_transform(
    *,
    height: int = 128,
    keep_aspect: bool = True,
    tight_crop_enabled: bool = False,
    tight_crop_threshold: int = 245,
    tight_crop_margin: int = 2,
    to_float_tensor: bool = True,
) -> Callable[[Image.Image], "object"]:
    crop = TightCrop(
        enabled=bool(tight_crop_enabled),
        threshold=int(tight_crop_threshold),
        margin=int(tight_crop_margin),
    )
    resize = ResizeToHeight(height=int(height), keep_aspect=bool(keep_aspect))

    def _pil_only(img: Image.Image) -> Image.Image:
        img = crop(img)
        img = resize(img)
        return img

    if not to_float_tensor:
        return _pil_only

    def _to_tensor(img: Image.Image):
        img = _pil_only(img)
        t = pil_to_tensor(img)  # uint8, [1,H,W]
        return t.to(dtype=torch.float32) / 255.0

    return _to_tensor
