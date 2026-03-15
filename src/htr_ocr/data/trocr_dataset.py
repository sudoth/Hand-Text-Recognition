from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


class TrOCRLineDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform: Callable[[Image.Image], Any] | None = None) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        text = str(row["text"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)

        image = image.convert("RGB")

        return {
            "image": image,
            "text": text,
            "image_path": str(image_path),
        }


def build_trocr_collate(processor, max_target_length: int):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        image_paths = [x["image_path"] for x in batch]

        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        tokenized = processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_target_length),
            return_tensors="pt",
        )
        labels = tokenized.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "texts": texts,
            "image_paths": image_paths,
        }

    return _collate