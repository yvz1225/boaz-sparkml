"""공식 Stanford IMDb 데이터에서 재현 가능한 균형 표본 CSV를 만든다."""

from __future__ import annotations

import argparse
import csv
import html
import random
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path


DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def clean_review(raw: bytes) -> str:
    text = html.unescape(raw.decode("utf-8"))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def read_sample(
    archive: tarfile.TarFile,
    split: str,
    label_name: str,
    label: int,
    size: int,
    rng: random.Random,
) -> list[dict[str, str | int]]:
    prefix = f"aclImdb/{split}/{label_name}/"
    members = sorted(
        (member for member in archive.getmembers() if member.name.startswith(prefix) and member.isfile()),
        key=lambda member: member.name,
    )
    if size > len(members):
        raise ValueError(f"{split}/{label_name} 표본 {size}개를 요청했지만 {len(members)}개만 있습니다.")

    rows = []
    # gzip tar를 뒤로 반복 탐색하지 않도록 선택된 파일을 압축 내 순서로 읽는다.
    for member in sorted(rng.sample(members, size), key=lambda item: item.offset_data):
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"리뷰를 읽을 수 없습니다: {member.name}")
        rows.append({"text": clean_review(extracted.read()), "label": label, "split": split})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stanford IMDb 균형 표본 생성")
    parser.add_argument("--output", default="/app/data/MovieReviews.csv")
    parser.add_argument("--train-per-label", type=int, default=500)
    parser.add_argument("--test-per-label", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"이미 존재합니다: {output} (--force로 다시 생성)")
        return

    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "aclImdb_v1.tar.gz"
        print(f"다운로드: {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, archive_path)

        rows = []
        with tarfile.open(archive_path, "r:gz") as archive:
            for split, size in (
                ("train", args.train_per_label),
                ("test", args.test_per_label),
            ):
                rows.extend(read_sample(archive, split, "neg", 0, size, rng))
                rows.extend(read_sample(archive, split, "pos", 1, size, rng))

    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"생성 완료: {output} "
        f"(train={args.train_per_label * 2}, test={args.test_per_label * 2})"
    )


if __name__ == "__main__":
    main()
