# GPU-Friendly Orin Nano Run

This branch keeps the second repo usable on Jetson/Orin Nano without forcing CPU or saving frame images by default.

## Environment

Use the NVIDIA-provided Jetson PyTorch/TorchVision install first, then:

```bash
cd ~/truckpipline_with_size/repo_size
bash scripts/setup_orin_nano.sh
```

## Run

```bash
cd ~/truckpipline_with_size/repo_size
bash scripts/run_auto_select_orin.sh /absolute/path/to/video.mp4
```

Current edge-safe defaults in this branch:
- `--device auto`
- `--save-frames 0`

If you explicitly want saved frame crops:

```bash
bash scripts/run_auto_select_orin.sh /absolute/path/to/video.mp4 --save-frames 1 --write-summary-csv
```
