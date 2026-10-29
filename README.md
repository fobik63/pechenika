# Cookie clicker detector

Windows screen clicker with YOLO detection. The app expects a four-class ONNX
model at `weights/best.onnx`: `game`, `ads_close`, `upgrade`, `bonus` in that
order. Ad close buttons and bonuses take priority over upgrades and the main
cookie. Each class has its own click cooldown. Press **Space** to stop; moving
the cursor to a screen corner also triggers PyAutoGUI's fail-safe.

## Run

Install Python 3.12, then:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py --confidence 0.45 --upgrade-cooldown 2
```

The supplied `weights/best.onnx` is an older **one-class** model. The app
rejects it until a four-class model is exported, preventing unsafe clicks.

## Train on local data

The screenshot and label directories are kept outside Git tracking but may
remain in the local checkout. Put matched images and YOLO labels in
`images/{train,val}` and `labels/{train,val}`. Use the class IDs in
`data.yaml`, including real examples of `bonus` in both splits. Fix the
mismatched image/label pairs reported by the validator. Run
`./prepare_dataset.ps1` after adding labels to make a group-aware 80/20 split;
unpaired files are preserved in `dataset_unmatched`. The current local split
has 295 pairs and no overlap between screenshot groups, but no `bonus` labels.

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt
.venv\Scripts\python validate_data.py
.venv\Scripts\python train.py
```

`train.py` validates all four classes, trains YOLO, and copies the exported
best ONNX model to `weights/best.onnx`. Training output and the private
dataset stay ignored by Git. Check detections on the actual game before
leaving the clicker unattended.
