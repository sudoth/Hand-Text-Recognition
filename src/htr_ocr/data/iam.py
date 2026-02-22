from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class IamLineRecord:
    line_id: str
    status: str
    graylevel: int
    n_components: int
    x: int
    y: int
    w: int
    h: int
    text: str


def parse_lines_txt(lines_txt_path: str | Path, keep_status: list[str] | None = None) -> list[IamLineRecord]:
    """
    Формат:
      line_id status graylevel n_components x y w h transcription

    transcription в IAM часто использует '|' вместо пробела.
    """
    keep_status = keep_status or ["ok"]

    records: list[IamLineRecord] = []
    with Path(lines_txt_path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 9:
                continue

            line_id = parts[0]
            status = parts[1]
            if status not in keep_status:
                continue

            graylevel = int(parts[2])
            n_components = int(parts[3])
            x, y, w, h = map(int, parts[4:8])

            text = " ".join(parts[8:]).replace("|", " ").strip()

            records.append(
                IamLineRecord(
                    line_id=line_id,
                    status=status,
                    graylevel=graylevel,
                    n_components=n_components,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    text=text,
                )
            )
    return records


def parse_forms_txt(forms_txt_path: str | Path) -> dict[str, str]:
    """
    Формат (пример из IAM):
      a01-000u 000 2 prt 7 5 52 36

    Нам нужен только (form_id, writer_id). Парсер tolerant:
    - пропускает комментарии '#'
    - берёт первые 2 токена.
    """
    mapping: dict[str, str] = {}
    p = Path(forms_txt_path)
    if not p.exists():
        return mapping

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def index_line_images(images_root: str | Path, exts: Iterable[str] = (".png", ".jpg", ".jpeg")) -> dict[str, Path]:
    """
    IAM может быть разложен по папкам по-разному (lines/a01/a01-000u/...),
    поэтому вместо жёсткого правила мы один раз индексируем все картинки.
    """
    images_root = Path(images_root)
    mapping: dict[str, Path] = {}

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    exts_l = {e.lower() for e in exts}

    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts_l:
            continue
        stem = p.stem
        # В IAM stem обычно уникален. Если встретились дубли — оставим первый.
        mapping.setdefault(stem, p)

    return mapping


def build_manifest(
    raw_dir: str | Path,
    images_subdir: str,
    annotations_path: str | Path,
    forms_path: str | Path | None,
    keep_status: list[str] | None,
    limit: int = 0,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
      line_id, form_id, writer_id (если есть), image_path, text, width, height, ...

    images_subdir — путь относительно raw_dir, где лежат изображения строк.
    """
    raw_dir = Path(raw_dir)
    images_root = raw_dir / images_subdir

    records = parse_lines_txt(annotations_path, keep_status=keep_status)
    if limit and limit > 0:
        records = records[:limit]

    # Индексация картинок
    img_index = index_line_images(images_root)

    # writer_id (если forms.txt доступен)
    forms_map: dict[str, str] = {}
    if forms_path is not None:
        forms_map = parse_forms_txt(forms_path)

    rows: list[dict] = []

    for rec in tqdm(records, desc="Building manifest"):
        img_path = img_index.get(rec.line_id)
        if img_path is None:
            raise FileNotFoundError(
                f"Image not found for line_id={rec.line_id}. "
                f"Searched under images_root={images_root} by filename stem."
            )

        with Image.open(img_path) as im:
            w_img, h_img = im.size

        form_id = "-".join(rec.line_id.split("-")[:2])
        writer_id = forms_map.get(form_id)

        rows.append(
            {
                "line_id": rec.line_id,
                "form_id": form_id,
                "writer_id": writer_id,
                "image_path": str(img_path),
                "text": rec.text,
                "status": rec.status,
                "width": w_img,
                "height": h_img,
                "graylevel": rec.graylevel,
                "n_components": rec.n_components,
                "bbox_x": rec.x,
                "bbox_y": rec.y,
                "bbox_w": rec.w,
                "bbox_h": rec.h,
            }
        )

    df = pd.DataFrame(rows)
    return df
