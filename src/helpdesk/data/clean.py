"""Clean raw tickets: drop near-duplicate messages (case and punctuation ignored)."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/tickets.parquet")
OUTPUT_PATH = Path("data/processed/tickets.parquet")


def clean(raw_path: Path = RAW_PATH, output_path: Path = OUTPUT_PATH) -> Path:
    df = pd.read_parquet(raw_path)
    before = len(df)
    key = (
        df["instruction"]
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.split()
        .str.join(" ")
    )
    df = df.loc[~key.duplicated()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dropped {before - len(df)} duplicates; {len(df)} rows left")
    return output_path


if __name__ == "__main__":
    clean()
