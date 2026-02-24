from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class LineSample:
    image_path: str
    text: str
    line_id: str | None = None
    form_id: str | None = None
    writer_id: str | None = None
    width: int | None = None
    height: int | None = None


class IamLineDataset:
    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable[[Image.Image], Any] | None = None,
        target_height: int = 128,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        
        required_columns = {"image_path", "text", "width", "height", "line_id", "form_id", "writer_id"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"CSV only contains columns: {list(self.df.columns)}")

        self.transform = transform
        self.target_height = int(target_height)

        w = self.df["width"].astype(float)
        h = self.df["height"].astype(float).replace(0, 1.0)
        self._approx_resized_width = (w * (self.target_height / h)).round().astype(int).clip(lower=1)

    def __len__(self) -> int:
        return int(len(self.df))

    def approx_resized_width(self, idx: int) -> int | None:
        if self._approx_resized_width is None:
            return None
        return int(self._approx_resized_width.iloc[idx])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[int(idx)]
        image_path = str(row["image_path"])
        text = str(row["text"])

        with Image.open(image_path) as im:
            im = im.convert("L")
            sample_img = im.copy()

        if self.transform is not None:
            pixel_values = self.transform(sample_img)
        else:
            pixel_values = sample_img

        out: dict[str, Any] = {
            "pixel_values": pixel_values,
            "text": text,
            "image_path": image_path,
        }

        for col in ["line_id", "form_id", "writer_id", "width", "height"]:
            if col in self.df.columns:
                out[col] = row[col]

        return out
