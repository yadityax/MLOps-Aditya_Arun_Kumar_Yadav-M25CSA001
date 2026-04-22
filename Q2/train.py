"""
Q2 Training Script — CityScape Image Segmentation (UNet, 23 classes)

Usage:
    # Step 1: download dataset (run once)
    gdown --folder 1GNe3Tu8Mud_CSLOiQZYHS2Rjq2sS74b_ -O data --remaining-ok

    # Step 2: train
    python train.py
"""

import os
import glob
import json
import subprocess
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import UNet

# ── Config ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 23
IMG_W, IMG_H = 128, 96          # cv2.resize takes (width, height)
EPOCHS      = 20                # ≥ 15 as required
BATCH_SIZE  = 8
LR          = 3e-4
DATA_ROOT   = "data"
Q2_DIR      = "Question2"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────


# ── Dataset ──────────────────────────────────────────────────────────────────
class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read Image (exact snippet from exam)
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        # Read Mask (exact snippet from exam)
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)
        # Clip to valid class range [0, NUM_CLASSES-1]
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

        img  = torch.from_numpy(img).permute(2, 0, 1)   # [3, H, W]
        mask = torch.from_numpy(mask).long()              # [H, W]
        return img, mask


# ── Combined Loss ────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)                         # [B, C, H, W]
        one_hot = F.one_hot(targets, self.num_classes)             # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()             # [B, C, H, W]
        inter   = (probs * one_hot).sum(dim=(2, 3))
        denom   = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice    = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(preds_flat, targets_flat, num_classes=NUM_CLASSES):
    """mIOU and mDice over flattened tensors (entire dataset at once)."""
    iou_list, dice_list = [], []
    for cls in range(num_classes):
        p = preds_flat == cls
        t = targets_flat == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ps = p.sum().item()
        ts = t.sum().item()
        iou_list.append(inter / union)
        dice_list.append(2 * inter / (ps + ts) if (ps + ts) > 0 else 0.0)
    miou  = float(np.mean(iou_list))  if iou_list  else 0.0
    mdice = float(np.mean(dice_list)) if dice_list else 0.0
    return miou, mdice


def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu()
            all_preds.append(preds.flatten())
            all_targets.append(masks.flatten())
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(Q2_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Download dataset if missing ──────────────────────────────────────────
    cam_rgb = os.path.join(DATA_ROOT, "CameraRGB")
    if not os.path.isdir(cam_rgb):
        print("CameraRGB folder not found — downloading dataset with gdown …")
        subprocess.run([
            "gdown", "--folder",
            "1GNe3Tu8Mud_CSLOiQZYHS2Rjq2sS74b_",
            "-O", DATA_ROOT,
        ], check=True)

    # ── Discover files ───────────────────────────────────────────────────────
    img_paths  = sorted(
        glob.glob(os.path.join(DATA_ROOT, "CameraRGB",  "*.png")) +
        glob.glob(os.path.join(DATA_ROOT, "CameraRGB",  "*.jpg"))
    )
    mask_paths = sorted(
        glob.glob(os.path.join(DATA_ROOT, "CameraMask", "*.png")) +
        glob.glob(os.path.join(DATA_ROOT, "CameraMask", "*.jpg"))
    )
    assert len(img_paths) > 0, "No images found in data/CameraRGB/"
    assert len(img_paths) == len(mask_paths), (
        f"Image/mask count mismatch: {len(img_paths)} vs {len(mask_paths)}"
    )
    print(f"Total samples: {len(img_paths)}")

    # ── 80/20 split, seed=42 ─────────────────────────────────────────────────
    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )
    print(f"Train: {len(tr_imgs)}  |  Test: {len(te_imgs)}")

    # Save test paths so the Gradio app can look up ground-truth masks by filename
    with open(os.path.join(Q2_DIR, "test_paths.json"), "w") as f:
        json.dump({"images": te_imgs, "masks": te_masks}, f, indent=2)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_ds = CityscapesDataset(tr_imgs, tr_masks)
    test_ds  = CityscapesDataset(te_imgs, te_masks)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=(DEVICE == "cuda"))
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=(DEVICE == "cuda"))

    # ── Model + optimiser ────────────────────────────────────────────────────
    model     = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    ce_crit   = nn.CrossEntropyLoss()
    dice_crit = DiceLoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ────────────────────────────────────────────────────────
    train_losses, miou_scores, mdice_scores = [], [], []
    best_miou   = 0.0
    best_mdice  = 0.0
    best_epoch  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = 0.5 * ce_crit(out, masks) + 0.5 * dice_crit(out, masks)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * imgs.size(0)
        scheduler.step()

        avg_loss            = ep_loss / len(train_ds)
        epoch_miou, epoch_mdice = evaluate(model, test_dl)
        train_losses.append(avg_loss)
        miou_scores.append(epoch_miou)
        mdice_scores.append(epoch_mdice)

        # Save best model
        if epoch_miou > best_miou:
            best_miou  = epoch_miou
            best_mdice = epoch_mdice
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(Q2_DIR, "best_model.pth"))

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  mIOU={epoch_miou:.4f}  mDice={epoch_mdice:.4f}"
            + ("  ← best" if epoch == best_epoch else "")
        )

    print(f"\n{'='*55}")
    print(f"Best Epoch : {best_epoch}")
    print(f"Test mIOU  : {best_miou:.4f}")
    print(f"Test mDice : {best_mdice:.4f}")
    print(f"{'='*55}\n")

    # ── Save metrics JSON ────────────────────────────────────────────────────
    metrics = {
        "test_miou":    best_miou,
        "test_mdice":   best_mdice,
        "best_epoch":   best_epoch,
        "epochs":       EPOCHS,
        "train_losses": [float(v) for v in train_losses],
        "miou_scores":  [float(v) for v in miou_scores],
        "mdice_scores": [float(v) for v in mdice_scores],
    }
    with open(os.path.join(Q2_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Save plots ───────────────────────────────────────────────────────────
    ep_axis = list(range(1, EPOCHS + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep_axis, train_losses, marker="o", linewidth=1.5)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss (CE + Dice)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(Q2_DIR, "training_loss.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep_axis, miou_scores, marker="o", color="steelblue", linewidth=1.5)
    ax.axhline(0.48, color="red", linestyle="--", linewidth=1, label="threshold 0.48")
    ax.set(xlabel="Epoch", ylabel="mIOU", title="mIOU over Epochs")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(Q2_DIR, "miou.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep_axis, mdice_scores, marker="o", color="darkorange", linewidth=1.5)
    ax.axhline(0.48, color="red", linestyle="--", linewidth=1, label="threshold 0.48")
    ax.set(xlabel="Epoch", ylabel="mDice", title="mDice over Epochs")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(Q2_DIR, "mdice.png"), dpi=150)
    plt.close(fig)

    print("Plots saved to Question2/")
    print(f"\nQuestion2: mIOU: {best_miou:.4f} and mDICE: {best_mdice:.4f}")
    print("(Copy the line above into your README.md)")


if __name__ == "__main__":
    main()
