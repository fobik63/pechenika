# Cookie clicker detector

Windows clicker using `weights/best.onnx`. The model has exactly three classes:
`game`, `ads_close`, `upgrade` (IDs 0, 1, 2). An ad close button has priority,
then an available upgrade, then the main cookie. A large ad detection is clicked
near its upper right corner; a small detection is clicked in its center.
Before buying, the app checks the shop's price state in the screenshot: green
prices are affordable, red prices are skipped. It randomly picks one affordable
row and keeps farming when none is available. The pixel check follows the
default Cookie Clicker Old shop colors shown in the local screenshots.

## Run

Install Python 3.12, then in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py --confidence 0.4 --upgrade-cooldown 4
```

**Space** pauses or resumes. **Ctrl+C** exits. Moving the mouse to a screen
corner activates PyAutoGUI's fail-safe. Use `--monitor 0` for all displays or
`--monitor 1` (default) for the first display. The app fails with a clear error
if the ONNX model is missing or has the wrong classes.

## Retrain locally

Training images and YOLO labels stay outside Git. Store matching pairs under
`images/{train,val}` and `labels/{train,val}` using IDs from `data.yaml`.
`prepare_dataset.ps1` rebuilds a group-aware 80/20 split while preserving
unpaired files in `dataset_unmatched`.

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt
./prepare_dataset.ps1
.venv\Scripts\python validate_data.py
.venv\Scripts\python train.py --epochs 100
```

The training script exports its best checkpoint to `weights/best.onnx`. Only
this deployment model is tracked in Git; local datasets, training runs, and
Python environments are ignored.
