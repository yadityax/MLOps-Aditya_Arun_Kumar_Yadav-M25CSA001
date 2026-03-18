# Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna
### English → Hindi Neural Machine Translation
**Student:** Aditya Arun Kumar Yadav &nbsp;|&nbsp; **Roll No:** M25CSA001 &nbsp;|&nbsp; 

---

##  Overview

This assignment optimizes a from-scratch PyTorch Transformer model for English→Hindi translation using **Ray Tune** paired with the **Optuna** search algorithm. The goal was to match or exceed the baseline BLEU score in significantly fewer epochs.

---

##  Repository Structure

```
├── M25CSA001_Ass_4_tuned_en_to_hi.ipynb   # Main tuning notebook (Ray Tune + Optuna)
├── M25CSA001_ass_4_report.pdf              # Assignment report (1-2 pages)
├── README.md                               # This file
└── .gitignore
```

> **Model Weights** are hosted on HuggingFace due to GitHub storage limits:
> 🔗 [M25CSA001/transformer-en-hi](https://huggingface.co/m25csa001/transformer-en-hi/tree/main)

---

##  Results Summary

| Metric | Baseline (100 epochs) | Tuned Model (40 epochs) | Improvement |
|--------|----------------------|-------------------------|-------------|
| BLEU Score | 52.47 | **73.70** | +21.23 points |
| Final Loss | 0.0974 | **0.0805** | 17% lower |
| Training Time | ~52 minutes | **4.4 minutes** | ~92% faster |
| Epochs Required | 100 | **40** | 60% fewer |

---

##  Best Hyperparameter Configuration

Found by **OptunaSearch** after 8 trials (14.5 minutes sweep):

| Hyperparameter | Baseline | Best Tuned Value |
|----------------|----------|-----------------|
| Learning Rate | 1e-4 | **2.037e-4** |
| Batch Size | 60 | **64** |
| Num Attention Heads | 8 | **8** |
| FFN Dimension (d_ff) | 2048 | **2048** |
| Dropout | 0.1 | **0.102** |
| Num Layers | 6 | **4** |
| d_model | 512 | 512 (fixed) |

---

##  Model Architecture

Custom Transformer built from scratch in PyTorch:
- Multi-Head Self Attention + Cross Attention
- Positional Encoding
- Layer Normalization
- Feed-Forward Network with ReLU
- Source & Target Padding Masks + Causal Mask

---

##  Tuning Setup

| Setting | Value |
|---------|-------|
| Search Algorithm | OptunaSearch (TPE, minimize loss) |
| Scheduler | ASHAScheduler (max_t=40, grace_period=2) |
| Number of Trials | 8 |
| Max Epochs per Trial | 40 |
| Concurrent Trials | 1 |
| GPU | RTX (1 GPU per trial) |
| Additional Tricks | OneCycleLR warm-up, Gradient Clipping (1.0) |

---

##  Sample Translations

| English | Hindi (Predicted) |
|---------|------------------|
| I love you. | मैं तुमसे प्यार करता हूँ। |
| What is your name? | आपका नाम क्या है? |
| How are you? | आप कैसे हो? |
| The weather is nice today. | मौसम आज मौसम है। |
| She is a good teacher. | वह एक अच्छा शिक्षक है। |

---

##  How to Run

### 1. Install Dependencies
```bash
pip install ray[tune] optuna torch nltk pandas tqdm huggingface_hub
```

### 2. Load the Trained Model
```python
import torch
from huggingface_hub import hf_hub_download

# Download model weights from HuggingFace
path = hf_hub_download(
    repo_id="M25CSA001/transformer-en-hi",
    filename="M25CSA001_Ass_4_best_model.pth"
)

# Load weights
model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()
```

### 3. Run the Full Notebook
Open `M25CSA001_Ass_4_tuned_en_to_hi.ipynb` and run all cells sequentially.
Make sure `English-Hindi.tsv` is in the same directory.

---

## 📦 Dependencies

```
torch
ray[tune]
optuna
nltk
pandas
tqdm
huggingface_hub
```

---
