"""Stanford IMDb 표본 CSV의 최소 무결성 검사."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="/app/data/MovieReviews.csv")
    path = Path(parser.parse_args().path)

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["text", "label", "split"]:
            raise ValueError(f"헤더는 text,label,split이어야 합니다: {reader.fieldnames}")
        rows = list(reader)

    if any(
        not row["text"].strip()
        or row["label"] not in {"0", "1"}
        or row["split"] not in {"train", "test"}
        for row in rows
    ):
        raise ValueError("빈 text, 잘못된 label 또는 잘못된 split이 있습니다.")
    if len({row["text"] for row in rows}) != len(rows):
        raise ValueError("중복 리뷰가 있습니다.")

    counts = Counter((row["split"], row["label"]) for row in rows)
    for split in ("train", "test"):
        if counts[(split, "0")] != counts[(split, "1")]:
            raise ValueError(f"{split} label이 균형을 이루지 않습니다: {dict(counts)}")
        if counts[(split, "0")] < 100:
            raise ValueError(f"{split}의 클래스별 데이터가 100개 미만입니다: {dict(counts)}")

    print(f"데이터 검사 통과: {path} ({len(rows)}행)")
    print(f"train: Negative {counts[('train', '0')]}, Positive {counts[('train', '1')]}")
    print(f"test: Negative {counts[('test', '0')]}, Positive {counts[('test', '1')]}")


if __name__ == "__main__":
    main()
