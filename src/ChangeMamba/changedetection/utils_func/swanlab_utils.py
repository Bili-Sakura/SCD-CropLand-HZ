"""SwanLab experiment tracking utilities for ChangeMamba training."""

import os

from ChangeMamba.changedetection.utils_func.project_env import load_project_dotenv

load_project_dotenv()

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    swanlab = None

_swanlab_active = False


def _serialize_config(obj):
    """Convert config to JSON-serializable format."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize_config(x) for x in obj[:100]]  # Limit list length
    if isinstance(obj, dict):
        return {str(k): _serialize_config(v) for k, v in list(obj.items())[:50]}
    return str(obj)


def init_swanlab(project="ChangeMamba", experiment_name=None, config=None, **kwargs):
    """Initialize SwanLab experiment. No-op if swanlab is not installed or SWANLAB_API_KEY is unset."""
    global _swanlab_active
    _swanlab_active = False
    if not SWANLAB_AVAILABLE:
        return None
    if not os.environ.get("SWANLAB_API_KEY"):
        return None
    cfg = config or {}
    if isinstance(cfg, dict):
        cfg = {k: _serialize_config(v) for k, v in cfg.items()}
    init_kwargs = {
        "project": project,
        "config": cfg,
        **kwargs
    }
    if experiment_name:
        init_kwargs["experiment_name"] = experiment_name
    out = swanlab.init(**init_kwargs)
    _swanlab_active = True
    return out


def log_metrics(metrics, step=None):
    """Log metrics to SwanLab. No-op if swanlab is not installed or init was skipped."""
    if not SWANLAB_AVAILABLE or swanlab is None or not _swanlab_active:
        return
    kwargs = {}
    if step is not None:
        kwargs["step"] = step
    swanlab.log(metrics, **kwargs)


def finish_swanlab():
    """Finish SwanLab experiment."""
    global _swanlab_active
    if not SWANLAB_AVAILABLE or swanlab is None or not _swanlab_active:
        return
    swanlab.finish()
    _swanlab_active = False
