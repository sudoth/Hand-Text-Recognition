from pathlib import Path
import random

import pandas as pd


def _normalize_fracs(train: float, val: float, test: float) -> tuple[float, float, float]:
    s = train + val + test
    if abs(s - 1.0) < 1e-6:
        return train, val, test
    return train / s, val / s, test / s


def make_group_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    train: float,
    val: float,
    test: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Детерминированный split на фиксированном сиде,
    далее распределяем группы по долям train/val/test.
    """
    train, val, test = _normalize_fracs(train, val, test)

    groups = df[group_col].dropna().unique().tolist()
    rnd = random.Random(int(seed))
    rnd.shuffle(groups)

    n = len(groups)
    n_train = int(round(train * n))
    n_val = int(round(val * n))

    g_train = set(groups[:n_train])
    g_val = set(groups[n_train : n_train + n_val])
    g_test = set(groups[n_train + n_val :])

    df_train = df[df[group_col].isin(g_train)].reset_index(drop=True)
    df_val = df[df[group_col].isin(g_val)].reset_index(drop=True)
    df_test = df[df[group_col].isin(g_test)].reset_index(drop=True)

    return df_train, df_val, df_test


def load_forms_mapping(forms_txt_path: str | Path) -> dict[str, str]:
    """
    Парсим forms.txt в dict[str, str]
    Формат: a01-000u 000 2 prt 7 5 52 36

    Нужен только (form_id, writer_id) для правильного сплита.
    Берём первые 2 токена и пропускаем комментарии.
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


def attach_writer_id(df: pd.DataFrame, forms_txt_path: str | Path) -> pd.DataFrame:
    mapping = load_forms_mapping(forms_txt_path)
    out = df.copy()
    out["writer_id"] = out["form_id"].map(mapping)
    return out
