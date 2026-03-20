"""Load repository-root ``.env`` via ``python-dotenv`` (listed in environment.yaml / ChangeMamba requirements)."""

from __future__ import annotations

from pathlib import Path

_loaded = False


def repo_root() -> Path:
    """Project root (directory containing ``example.env``)."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "example.env").is_file():
            return ancestor
    return here.parents[4]


def load_project_dotenv(*, override: bool = False) -> None:
    """Populate os.environ from ``<repo>/.env``. No-op if python-dotenv is missing or no file."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = repo_root() / ".env"
    if path.is_file():
        load_dotenv(path, override=override)
