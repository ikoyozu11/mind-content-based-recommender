import os
import pandas as pd

BASE_DIR = "datasets"
SPLITS = ["MINDsmall_train", "MINDsmall_dev"]
FILES = ["news.tsv", "behaviors.tsv"]

NEWS_COLS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities"
]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

def read_tsv_safely(path: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        na_filter=False,
        encoding="utf-8",
    )
    if kind == "news":
        df.columns = NEWS_COLS
    elif kind == "behaviors":
        df.columns = BEH_COLS
    return df

def convert(tsv_path: str) -> str:
    folder = os.path.dirname(tsv_path)
    fname = os.path.basename(tsv_path)
    stem = os.path.splitext(fname)[0]

    kind = "news" if stem == "news" else "behaviors"
    print(f"[+] Reading {tsv_path}")

    df = read_tsv_safely(tsv_path, kind=kind)

    out_csv = os.path.join(folder, f"{stem}.csv")
    print(f"    -> Writing CSV {out_csv}")

    df.to_csv(out_csv, index=False, encoding="utf-8")

    return out_csv

def main():
    converted = []
    for split in SPLITS:
        for f in FILES:
            tsv_path = os.path.join(BASE_DIR, split, f)
            if not os.path.exists(tsv_path):
                print(f"[!] Not found, skip: {tsv_path}")
                continue
            converted.append(convert(tsv_path))

    print("\n✅ Done. Converted files:")
    for p in converted:
        print(" -", p)

if __name__ == "__main__":
    main()