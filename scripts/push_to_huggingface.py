#!/usr/bin/env python3
"""Publish a local checkpoint folder to Hugging Face.

Default path uses **Storage Buckets** (S3-like, Xet-backed, no git history) via ``sync_bucket`` —
see https://huggingface.co/docs/huggingface_hub/en/guides/buckets

Optional ``--mode model`` keeps the classic **git/LFS model repo** ``upload_folder`` flow for
model cards on the Hub.

Auth: ``HF_TOKEN`` (write) or ``--token``: https://huggingface.co/settings/tokens.
Optional repo-root ``.env`` — copy from ``example.env``; loaded automatically via ``python-dotenv``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ChangeMamba.changedetection.utils_func.project_env import load_project_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


_DEFAULT_BUCKET_ID = "BiliSakura/ChangeMambaSCD-JL1-CUP-2024"
_DEFAULT_MODEL_REPO_ID = "BiliSakura/JL1-ChangeMambaSCD"

_SYNC_EXCLUDE = [
    "*.log",
    "*.tmp",
    ".DS_Store",
    ".git",
    ".gitignore",
    "*.pyc",
    "__pycache__",
]


def _bucket_dest(bucket_id: str, prefix: str | None) -> str:
    bid = bucket_id.strip().strip("/")
    base = f"hf://buckets/{bid}"
    if prefix:
        p = prefix.strip().strip("/")
        if p:
            return f"{base}/{p}"
    return base


def main() -> int:
    load_project_dotenv()
    root = _project_root()
    default_local = root / "models" / "BiliSakura" / "JL1-ChangeMambaSCD"

    p = argparse.ArgumentParser(
        description="Upload local artifacts to Hugging Face (buckets by default, or model repo)."
    )
    p.add_argument(
        "--mode",
        choices=("bucket", "model"),
        default=os.environ.get("HF_PUSH_MODE", "bucket"),
        help="bucket = Storage Buckets sync (default); model = git-based model repo",
    )
    p.add_argument(
        "--bucket-id",
        default=os.environ.get("HF_BUCKET_ID", _DEFAULT_BUCKET_ID),
        help="namespace/bucket_name for buckets mode (env HF_BUCKET_ID)",
    )
    p.add_argument(
        "--prefix",
        default=os.environ.get("HF_BUCKET_PREFIX") or None,
        help="Optional path inside the bucket, e.g. v1 or jl1 (env HF_BUCKET_PREFIX)",
    )
    p.add_argument(
        "--no-create-bucket",
        action="store_true",
        help="Do not call create_bucket; bucket must already exist",
    )
    p.add_argument(
        "--repo-id",
        default=os.environ.get("HF_MODEL_REPO_ID", _DEFAULT_MODEL_REPO_ID),
        help="Hub model repo id for --mode model (env HF_MODEL_REPO_ID)",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=default_local,
        help="Folder to upload (README.md, checkpoints, …)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF API token (default: env HF_TOKEN)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="When creating bucket/repo: make it private",
    )
    p.add_argument(
        "--no-create-repo",
        action="store_true",
        help="[model mode] Do not create_repo; repo must exist",
    )
    p.add_argument(
        "--commit-message",
        default="Update model weights and model card",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help="[bucket mode] Remove remote files not present locally (mirror)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="[bucket mode] Print sync plan only, no upload",
    )
    args = p.parse_args()

    local = args.local_dir.resolve()
    if not local.is_dir():
        print(f"ERROR: local dir not found: {local}", file=sys.stderr)
        return 1
    if not args.token:
        print(
            "ERROR: No token. Set HF_TOKEN or pass --token "
            "(https://huggingface.co/settings/tokens ).",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import HfApi, create_bucket
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)

    if args.mode == "bucket":
        bucket_id = args.bucket_id.strip()
        dest = _bucket_dest(bucket_id, args.prefix)
        if not args.no_create_bucket:
            create_bucket(
                bucket_id,
                private=args.private if args.private else None,
                exist_ok=True,
                token=args.token,
            )
        plan = api.sync_bucket(
            str(local),
            dest,
            delete=args.delete,
            dry_run=args.dry_run,
            exclude=_SYNC_EXCLUDE,
            token=args.token,
        )
        if args.dry_run:
            print(plan.summary())
        web = f"https://huggingface.co/buckets/{bucket_id}"
        print(f"Synced {local} -> {dest}  (browse: {web})")
        return 0

    # --mode model
    repo_id = args.repo_id.strip()
    if not args.no_create_repo:
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=[".git*", "*.tmp", "__pycache__/*", "*.log", ".DS_Store"],
    )
    print(f"Uploaded {local} -> https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
