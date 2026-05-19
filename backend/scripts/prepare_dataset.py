import json
import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/issues.jsonl")
OUTPUT_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"`{3}.*?`{3}", " CODE_BLOCK ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " CODE ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": int(row["id"]),
                "number": int(row["number"]),
                "text": row["text"],
                "label": row["label"],
                "created_at": row["created_at"],
                "url": row["url"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_per_label(df: pd.DataFrame):
    train_parts = []
    val_parts = []
    test_parts = []

    for label, group in df.groupby("label"):
        group = group.sort_values("created_at").reset_index(drop=True)

        n = len(group)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, val_df, test_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            issue = json.loads(line)

            text = clean_text(issue["title"] + "\n\n" + issue["body"])

            if len(text) < 30:
                continue

            rows.append(
                {
                    "id": issue["id"],
                    "number": issue["number"],
                    "text": text,
                    "label": issue["target_label"],
                    "created_at": issue["created_at"],
                    "url": issue["url"],
                }
            )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id"])
    df = df.sort_values("created_at").reset_index(drop=True)

    print("Total examples:", len(df))

    print("\nOverall label counts:")
    print(df["label"].value_counts())

    train_df, val_df, test_df = split_per_label(df)

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    print("\nTrain labels:")
    print(train_df["label"].value_counts())

    print("\nVal labels:")
    print(val_df["label"].value_counts())

    print("\nTest labels:")
    print(test_df["label"].value_counts())

    save_jsonl(train_df, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_df, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_df, OUTPUT_DIR / "test.jsonl")

    print("\nSaved processed dataset:")
    print("data/processed/train.jsonl")
    print("data/processed/val.jsonl")
    print("data/processed/test.jsonl")


if __name__ == "__main__":
    main()