# HTR Diplom: Transformers for Handwritten Text Recognition



```
./data/raw/
  ascii/
    forms.txt
    lines.txt
  lines/
    a01/...
    a02/...
    ...
```

Внутри `data/raw/lines/**` могут быть вложенные папки (например `a01/a01-000u/...`).
Скрипт `make_manifest` не зависит от конкретной вложенности: он индексирует все картинки и находит их по `line_id`.

## тлдр

### 1) Установка
```bash
uv sync
```

### 2) Сборка manifest
```bash
uv run htr make_manifest
```

Результат: `data/processed/manifest.parquet`.

### 3) Сплиты train/val/test

**Writer-independent (рекомендуется):**
```bash
uv run htr make_splits split.strategy=writer
```

**Form-independent:**
```bash
uv run htr make_splits split.strategy=form
```

Результат: `data/processed/train.csv`, `val.csv`, `test.csv`.

## CLI + Hydra overrides
Мы используем **Fire** как CLI оболочку, а параметры задаём через **Hydra overrides**:

`uv run htr <command> <override1> <override2> ...`
`uv run htr make_manifest data.raw_dir=data/raw preprocess.height=128`

## MLflow
MLflow конфиг ляжет в `configs/mlflow/local.yaml`. По умолчанию используется локальный трекинг (`./mlruns`).

### UI
```bash
uv run mlflow ui
```