from typing import Any
import torch
import torch.nn.functional as F


def collate_line_batch(batch: list[dict[str, Any]], pad_value: float = 1.0) -> dict[str, Any]:
    """
    Аргументы:
    pixel_values: torch.FloatTensor [1, H, W] (значения из [0,1])
    text: str

    Возвращает:
    pixel_values: [B, 1, H, Wmax]
    pixel_mask:  [B, Wmax] (1 для настоящих, 0 для паддинга)
    """

    if not batch:
        raise ValueError("Empty batch")
    
    imgs = [b["pixel_values"] for b in batch]
    texts = [str(b.get("text", "")) for b in batch]

    widths = [int(img.shape[-1]) for img in imgs]
    heights = [int(img.shape[-2]) for img in imgs]
    if len(set(heights)) != 1:
        raise ValueError(f"Different images height. Current heights={sorted(set(heights))}")

    w_max = max(widths)
    padded = []
    masks = []

    for img, w in zip(imgs, widths, strict=True):
        pad_right = w_max - w
        if pad_right < 0:
            raise RuntimeError("Negative padding")
        if pad_right:
            img = F.pad(img, pad=(0, pad_right, 0, 0), mode="constant", value=float(pad_value))
        padded.append(img)
        mask = torch.zeros((w_max,), dtype=torch.bool)
        mask[:w] = True
        masks.append(mask)

    pixel_values = torch.stack(padded, dim=0)  # [B, 1, H, W]
    pixel_mask = torch.stack(masks, dim=0)  # [B, W]

    meta: list[dict[str, Any]] = []
    for b in batch:
        meta.append(
            {
                "line_id": b.get("line_id"),
                "form_id": b.get("form_id"),
                "writer_id": b.get("writer_id"),
                "image_path": b.get("image_path"),
            }
        )

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "texts": texts,
        "widths": widths,
        "meta": meta,
    }
