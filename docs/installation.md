# Installation

This project uses the **ChangeMamba** backbone and requires **rasterio** for GeoTIFF imagery. Installation supports either a pre-built environment or a manual setup.

---

## Requirements

- **OS**: Linux (recommended)
- **Python**: 3.10
- **CUDA**: 11.6+ (11.8 recommended)
- **GPU**: NVIDIA GPU with compute capability 7.0+

---

## Option A: Use Pre-built Environment (Recommended)

The `mambascd` environment is already configured with all dependencies. Use the provided environment files from the workspace root:

```bash
# From workspace root (sakura/)
mamba env create -f environment.yaml -n mambascd
# or, if the env already exists:
mamba env update -f environment.yaml -n mambascd
```

Then activate:

```bash
conda activate mambascd
```

---

## Option B: Manual Setup

### 1. Create environment

```bash
conda create -n mambascd python=3.10 -y
conda activate mambascd
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install ChangeMamba dependencies

```bash
cd src/ChangeMamba
pip install -r requirements.txt
```

### 4. Build selective_scan CUDA kernel

```bash
cd kernels/selective_scan
pip install .
cd ../../..
```

Verify the kernel:

```bash
pytest src/ChangeMamba/kernels/selective_scan/test_selective_scan.py
```

### 5. Install rasterio (for GeoTIFF handling)

```bash
pip install rasterio
```

### 5b. Install Prodigy optimizer (optional)

```bash
pip install prodigyopt
```

### 6. Optional: Detection / Segmentation dependencies

For downstream MMEngine-based tasks:

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

---

## Verify Installation

```bash
conda activate mambascd
python -c "import torch; import rasterio; print('OK')"
```
