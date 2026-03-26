#!/usr/bin/env python3
"""Write train/val/test metadata.csv for Hugging Face Dataset Viewer (one row per patch)."""

from __future__ import annotations

import argparse
from pathlib import Path


def stems_from_train_list(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        out.append(s if s.endswith(".png") else f"{s}.png")
    return out


def write_split_metadata(root: Path, split: str, stems: list[str], with_labels: bool) -> None:
    out_path = root / split / "metadata.csv"
    # Hugging Face joins each path with dirname(metadata.csv). Use paths relative to
    # train/ | val/ | test/ (e.g. T1/00001.png), not repo-root paths like train/T1/...
    if with_labels:
        cols = ["t1_file_name", "t2_file_name", "gt_t1_file_name", "gt_t2_file_name", "gt_cd_file_name"]
        lines = [",".join(cols)]
        for name in stems:
            base = name[:-4] if name.endswith(".png") else name
            fn = f"{base}.png"
            t1 = f"T1/{fn}"
            t2 = f"T2/{fn}"
            g1 = f"GT_T1/{fn}"
            g2 = f"GT_T2/{fn}"
            gcd = f"GT_CD/{fn}"
            lines.append(",".join([t1, t2, g1, g2, gcd]))
    else:
        cols = ["t1_file_name", "t2_file_name"]
        lines = [",".join(cols)]
        for name in stems:
            base = name[:-4] if name.endswith(".png") else name
            fn = f"{base}.png"
            lines.append(",".join([f"T1/{fn}", f"T2/{fn}"]))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(stems)} rows -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "JL1_second",
        help="JL1_second dataset root (contains train.txt, val/, …)",
    )
    args = p.parse_args()
    root = args.root
    train_stems = stems_from_train_list(root / "train.txt")
    val_stems = stems_from_train_list(root / "val.txt")
    test_stems = stems_from_train_list(root / "test.txt")
    write_split_metadata(root, "train", train_stems, with_labels=True)
    write_split_metadata(root, "val", val_stems, with_labels=True)
    write_split_metadata(root, "test", test_stems, with_labels=False)


if __name__ == "__main__":
    main()
