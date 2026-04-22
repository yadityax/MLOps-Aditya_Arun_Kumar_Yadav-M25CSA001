# Q2: CityScape Image Segmentation — End-to-End Pipeline

## Files

| Path | Description |
|---|---|
| `model.py` | UNet architecture (4-level encoder-decoder, 23-class output) |
| `train.py` | Training script — downloads data, trains UNet, saves plots + checkpoint |
| `app.py` | Gradio 2-page app (Page 1: plots & metrics, Page 2: segmentation inference) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | App container (CPU; GPU optional) |
| `docker-compose.yml` | Single-service stack |
| `Question2/` | Generated artefacts: plots, `best_model.pth`, `metrics.json` |

---

## Step-by-step

### 1 — Install dependencies
```bash
cd /home/m25csa001/MajorExam_Q2
pip install -r requirements.txt
```

### 2 — Download the dataset
```bash
gdown --folder 1GNe3Tu8Mud_CSLOiQZYHS2Rjq2sS74b_ -O data --remaining-ok
```
Expected layout after download:
```
data/
├── CameraRGB/    ← RGB input images
└── CameraMask/   ← ground-truth segmentation masks (23 classes)
```

### 3 — Train the model
```bash
python train.py
```
This will:
- Split data 80 % train / 20 % test (seed=42)
- Train UNet for 20 epochs (CE + Dice loss, AdamW + CosineAnnealing LR)
- Evaluate mIOU & mDice on the test set after every epoch
- Save `Question2/best_model.pth`, `Question2/metrics.json`
- Save `Question2/training_loss.png`, `Question2/miou.png`, `Question2/mdice.png`

### 4 — Run the Gradio app (locally)
```bash
python app.py
```
Open **http://localhost:7860**

### 5 — Run with Docker
```bash
# After training (Question2/ folder must exist)
docker compose up --build
```
Open **http://localhost:7860**

---

## Test Set Results

Question2: mIOU: 0.5457 and mDICE: 0.5841

---

## App Pages

**Page 1 — Training Results**
- Three training plots (Loss, mIOU, mDice curves)
- Test-set mIOU and mDice scores in a table

**Page 2 — Segmentation Inference**
- Upload 4 images from the test set
- Optionally upload 4 corresponding ground-truth masks
- App displays: Original · Ground Truth · Prediction side-by-side for each sample
