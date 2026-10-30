"""Download the Bitext customer-support dataset to data/raw/."""

from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
OUTPUT_PATH = Path("data/raw/tickets.parquet")


def ingest(output_path: Path = OUTPUT_PATH) -> Path:
    ds = load_dataset(DATASET_NAME, split="train", revision=DATASET_REVISION)
    df = ds.to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Rows: {len(df)}; columns: {df.columns.tolist()}")

    return output_path


if __name__ == "__main__":
    ingest()
