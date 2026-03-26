#!/usr/bin/env python3
"""
Rebuild cropland BCD collection manifests from already-converted tiles.

Use this when conversion finished but list files are missing/empty (e.g. JL1 raw source absent).

Writes (under collections_root):
  - train_stage1_jl1.txt
  - train_stage2_joint.txt
  - train_stage3_input_quick_trainval.txt
  - val_stage3_input_quick_val.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _iter_triplet_names(split_root: Path) -> list[str]:
    """Return sorted names present in T1/T2/GT under split_root."""
    t1 = split_root / "T1"
    t2 = split_root / "T2"
    gt = split_root / "GT"
    if not (t1.is_dir() and t2.is_dir() and gt.is_dir()):
        return []
    out: list[str] = []
    for p in sorted(t1.glob("*.png")):
        name = p.name
        if (t2 / name).is_file() and (gt / name).is_file():
            out.append(name)
    return out


def _write_list(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines:
            f.write(x.rstrip("\n") + "\n")


def rebuild_stage1_jl1(collections_root: Path) -> list[str]:
    src = collections_root / "jl1_competition"
    entries: list[str] = []
    for split in ("train", "val"):
        for name in _iter_triplet_names(src / split):
            entries.append(f"jl1_competition/{split}/{name}")
    return sorted(entries)


def rebuild_stage2_joint(collections_root: Path, include_hrscd: bool) -> list[str]:
    sources = ["clcd", "fpcd", "hi_cna"]
    if include_hrscd:
        sources.append("hrscd")

    entries: list[str] = []
    for source in sources:
        src = collections_root / source
        for split in ("train", "val", "test"):
            for name in _iter_triplet_names(src / split):
                entries.append(f"{source}/{split}/{name}")
    return sorted(entries)


def rebuild_stage3_input_quick(collections_root: Path) -> tuple[list[str], list[str]]:
    src = collections_root / "input_quick"
    train = [f"input_quick/train/{n}" for n in _iter_triplet_names(src / "train")]
    val = [f"input_quick/val/{n}" for n in _iter_triplet_names(src / "val")]
    trainval = sorted(train + val)
    val_only = sorted(val)
    return trainval, val_only


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--collections_root",
        type=Path,
        default=Path("datasets/cropland_bcd_collections"),
        help="Converted collections root (default: datasets/cropland_bcd_collections).",
    )
    ap.add_argument(
        "--include_hrscd",
        action="store_true",
        help="Include hrscd/* in train_stage2_joint.txt (default: off).",
    )
    args = ap.parse_args()
    root = args.collections_root.resolve()

    if not root.is_dir():
        raise SystemExit(f"collections_root not found: {root}")

    stage1 = rebuild_stage1_jl1(root)
    if not stage1:
        raise SystemExit(
            "train_stage1_jl1 would be empty. Expected tiles under "
            f"{root}/jl1_competition/{{train,val}}/{{T1,T2,GT}}/*.png"
        )
    _write_list(root / "train_stage1_jl1.txt", stage1)
    print(f"[STAGE1] train_stage1_jl1.txt: {len(stage1)}")

    stage2 = rebuild_stage2_joint(root, include_hrscd=bool(args.include_hrscd))
    if not stage2:
        raise SystemExit("train_stage2_joint would be empty (missing converted tiles?).")
    _write_list(root / "train_stage2_joint.txt", stage2)
    print(f"[STAGE2] train_stage2_joint.txt: {len(stage2)} include_hrscd={bool(args.include_hrscd)}")

    stage3_trainval, stage3_val = rebuild_stage3_input_quick(root)
    if not stage3_trainval or not stage3_val:
        raise SystemExit("Stage3 manifests would be empty (missing input_quick tiles?).")
    _write_list(root / "train_stage3_input_quick_trainval.txt", stage3_trainval)
    _write_list(root / "val_stage3_input_quick_val.txt", stage3_val)
    print(
        f"[STAGE3] train_stage3_input_quick_trainval.txt: {len(stage3_trainval)} "
        f"val_stage3_input_quick_val.txt: {len(stage3_val)}"
    )


if __name__ == "__main__":
    main()

