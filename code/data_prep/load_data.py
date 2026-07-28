"""Data loading utilities."""
from pathlib import Path
import pandas as pd
import hashlib

from code.utils.config import RAW_DIR, PROCESSED_DIR


def load_raw(path: Path = None) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    path = path or (RAW_DIR / "project_risk_raw_dataset.csv")
    return pd.read_csv(path)


def write_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a DataFrame to processed parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    p = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_processed(name: str) -> pd.DataFrame:
    """Load a processed parquet file."""
    p = PROCESSED_DIR / f"{name}.parquet"
    return pd.read_parquet(p)


def file_hash(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Compute file hash for integrity verification."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()