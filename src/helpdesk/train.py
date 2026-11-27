"""Train a baseline ticket classifier."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/processed/tickets.parquet")
TEXT_COL = "instruction"
SEED = 42
TEST_SIZE = 0.2
METRICS_PATH = Path("metrics.json")


def train(
    label_col: str = "category",
    ngram_max: int = 1,
    analyzer: str = "word",
    C: float = 1.0,
    class_weight: str | None = None,
) -> float:
    mlflow.set_experiment("helpdesk-baseline")
    with mlflow.start_run():
        df = pd.read_parquet(DATA_PATH)
        params = {
            "label_col": label_col,
            "ngram_max": ngram_max,
            "analyzer": analyzer,
            "C": C,
            "class_weight": class_weight,
            "dedup": "normalized",
            "test_size": TEST_SIZE,
            "seed": SEED,
            "n_rows": len(df),
        }
        mlflow.log_params(params)

        X, y = df[TEXT_COL], df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=(1, ngram_max))),
                ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        score = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("macro_f1", score)
        METRICS_PATH.write_text(json.dumps({"macro_f1": score}, indent=2) + "\n")
        mlflow.sklearn.log_model(model, name="model")
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="category", choices=["category", "intent"])
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--analyzer", default="word", choices=["word", "char_wb"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default=None, choices=["balanced"])
    args = parser.parse_args()

    score = train(
        label_col=args.label,
        ngram_max=args.ngram_max,
        analyzer=args.analyzer,
        C=args.C,
        class_weight=args.class_weight,
    )
    print(f"macro-F1: {score:.4f}")
