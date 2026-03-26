# Quick Start

```bash
# installed given ChangeMamba repo
# with additional rasterio, swanlab, python-dotenv
# Secrets and Hub defaults: copy example.env to .env and fill in (see example.env comments)
conda activate mambascd
```

## Troubleshooting: clear stuck Gradio GPU processes

If VRAM stays allocated after stopping `gradio_large_image_infer.py`, run:

```bash
pgrep -f "gradio_large_image_infer.py" | xargs -r kill -TERM
sleep 1
pgrep -f "gradio_large_image_infer.py" | xargs -r kill -KILL
nvidia-smi
```
