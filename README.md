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

## ТЛ;ДР

### 1) Установка
```bash
uv sync
```

### 2) Общий формат CLI
```bash
uv run htr <command> <override1> <override2> ...
```

### 3) Все команды CLI

`make_manifest`:
```bash
uv run htr make_manifest
uv run htr make_manifest data.raw_dir=data/raw data.limit=1000 data.keep_status='[ok,err]'
```
Результат: `data/processed/manifest.parquet`.

`make_splits`:
```bash
uv run htr make_splits split=writer
uv run htr make_splits split=form
uv run htr make_splits split=form split.seed=42 split.train=0.8 split.val=0.1 split.test=0.1
```
Результат: `data/processed/train.csv`, `val.csv`, `test.csv`.

`inspect_data`:
```bash
uv run htr inspect_data
uv run htr inspect_data loader.split=val loader.batch_size=4 loader.n_batches=2 loader.bucket.enabled=false preprocess.height=128
```

`inspect_augmentations`:
```bash
uv run htr inspect_augmentations
uv run htr inspect_augmentations inspect_aug.index=10 inspect_aug.n=16 inspect_aug.cols=4 loader.split=train augment=paper
uv run htr inspect_augmentations /abs/path/to/image.png inspect_aug.n=8
```

`train_crnn_ctc`:
```bash
uv run htr train_crnn_ctc
uv run htr train_crnn_ctc train.device=cpu train.epochs=20 train.lr=1e-3 loader.batch_size=8 preprocess.height=96
```

`eval_crnn_ctc`:
```bash
uv run htr eval_crnn_ctc
uv run htr eval_crnn_ctc eval.checkpoint_path=runs/crnn_ctc/best.pt eval.split=test eval.device=cpu decode=beam decode.beam_width=50 decode.topk=20
```

`infer_crnn_ctc`:
```bash
uv run htr infer_crnn_ctc
uv run htr infer_crnn_ctc infer.checkpoint_path=runs/crnn_ctc/best.pt infer.image_path=data/infer/image.png infer.device=cpu decode=greedy
```

## MLflow
Конфиг: `configs/mlflow/local.yaml`. По умолчанию локальный трекинг (`./mlruns`).

Отключить MLflow для любой команды:
```bash
uv run htr <command> mlflow.enabled=false
```

### UI
```bash
uv run mlflow ui
```
