"""
Data visualization utilities for remote sensing datasets.
Borrowed from CdSC (Cross-Difference Semantic Consistency Network).
"""

import numpy as np

try:
    from src.datasets.colormap import index2color
except ImportError:
    try:
        from datasets.colormap import index2color
    except ImportError:
        from ..datasets.colormap import index2color


def plot_img_and_mask(img: np.ndarray, mask: np.ndarray, colorize_mask: bool = True):
    """Plot input image and segmentation mask side by side.

    Args:
        img: HxWx3 image array
        mask: HxW mask - either class indices (int) or already RGB
        colorize_mask: If True and mask is index map, convert to RGB via colormap
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].set_title("Input image")
    axes[0].imshow(img)
    axes[0].axis("off")

    axes[1].set_title("Output mask")
    if colorize_mask and mask.ndim == 2 and np.issubdtype(mask.dtype, np.integer):
        mask_display = index2color(mask)
        axes[1].imshow(mask_display)
    else:
        axes[1].imshow(mask)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_img(img: np.ndarray, title: str = ""):
    """Display a single image."""
    import matplotlib.pyplot as plt

    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
