"""
Q2 Gradio App — CityScape Image Segmentation
2-page app:
  Page 1 · Training plots (loss, mIOU, mDice) + test-set metrics
  Page 2 · Upload 4 images (+optional masks) → show GT vs predicted masks

Run:
    python app.py
Then open http://localhost:7860
"""

import os
import json
import numpy as np
import cv2
import torch
import gradio as gr
from PIL import Image

from model import UNet

# ── Config ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 23
IMG_W, IMG_H = 128, 96      # same as training
Q2_DIR  = "Question2"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed colormap — same seed as training so colors are stable
np.random.seed(0)
COLORMAP = np.random.randint(30, 230, (NUM_CLASSES, 3), dtype=np.uint8)
# ─────────────────────────────────────────────────────────────────────────────


# ── Load model ───────────────────────────────────────────────────────────────
ckpt_path = os.path.join(Q2_DIR, "best_model.pth")
assert os.path.isfile(ckpt_path), (
    f"Checkpoint not found: {ckpt_path}\n"
    "Run  python train.py  first to train the model."
)
model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()
print(f"Model loaded from '{ckpt_path}' on {DEVICE}")


# ── Load saved metrics ────────────────────────────────────────────────────────
with open(os.path.join(Q2_DIR, "metrics.json")) as f:
    METRICS = json.load(f)
TEST_MIOU  = METRICS["test_miou"]
TEST_MDICE = METRICS["test_mdice"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    """Convert (H, W) class-index array → RGB PIL image."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        rgb[mask_np == cls] = COLORMAP[cls]
    return Image.fromarray(rgb)


def run_model(img_np: np.ndarray) -> np.ndarray:
    """Preprocess + forward pass; returns (H, W) class-index array."""
    img_r = cv2.resize(img_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    tensor = (
        torch.from_numpy(img_r.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )
    with torch.no_grad():
        return model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()


def mask_file_to_classes(mask_np: np.ndarray) -> np.ndarray:
    """Convert raw uploaded mask (H, W, 3) → (H, W) class indices."""
    m = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    return np.clip(np.max(m, axis=-1), 0, NUM_CLASSES - 1)


def auto_mask_path(img_filepath: str) -> str | None:
    """Given an uploaded image filepath, return the corresponding CameraMask path if it exists."""
    if img_filepath is None:
        return None
    fname = os.path.basename(img_filepath)
    # Try relative to cwd first, then absolute
    for base in [os.getcwd(), os.path.dirname(img_filepath)]:
        candidate = os.path.join(base, "data", "CameraMask", fname)
        if os.path.isfile(candidate):
            return candidate
    return None


def process_pair(img_filepath):
    """
    Given an uploaded image filepath, return
    (original PIL, ground-truth PIL or None, prediction PIL).
    """
    if img_filepath is None:
        return None, None, None

    img_np = cv2.imread(img_filepath)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    orig_pil = Image.fromarray(
        cv2.resize(img_np, (IMG_W * 3, IMG_H * 3), interpolation=cv2.INTER_NEAREST)
    )
    pred_idx = run_model(img_np)
    pred_pil = colorize_mask(pred_idx)

    gt_pil = None
    mask_path = auto_mask_path(img_filepath)
    if mask_path:
        mask_np = cv2.imread(mask_path)
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2RGB)
        gt_cls = mask_file_to_classes(mask_np)
        gt_pil = colorize_mask(gt_cls)

    return orig_pil, gt_pil, pred_pil


def infer_all(img1, img2, img3, img4):
    out = []
    for img in [img1, img2, img3, img4]:
        o, g, p = process_pair(img)
        out.extend([o, g, p])
    return out


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="CityScape Segmentation — Q2") as demo:

    gr.Markdown("# CityScape Image Segmentation — Q2")

    with gr.Tabs():

        # ── Page 1: Training Results ──────────────────────────────────────────
        with gr.Tab("Page 1 · Training Results"):
            gr.Markdown("## Training Curves")
            with gr.Row():
                gr.Image(
                    os.path.join(Q2_DIR, "training_loss.png"),
                    label="Training Loss",

                )
                gr.Image(
                    os.path.join(Q2_DIR, "miou.png"),
                    label="mIOU over Epochs",

                )
                gr.Image(
                    os.path.join(Q2_DIR, "mdice.png"),
                    label="mDice over Epochs",

                )
            gr.Markdown(
                f"## Test Set Metrics\n\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| **mIOU** | `{TEST_MIOU:.4f}` |\n"
                f"| **mDice** | `{TEST_MDICE:.4f}` |"
            )

        # ── Page 2: Segmentation Inference ───────────────────────────────────
        with gr.Tab("Page 2 · Segmentation Inference"):
            gr.Markdown(
                "Upload **4 images** from the test set (`data/CameraRGB/`).  \n"
                "The app shows: **Original · Ground Truth · Prediction** for each sample.  \n"
                "Ground-truth masks are loaded automatically from `data/CameraMask/`."
            )

            gr.Markdown(
                "> Ground-truth masks are loaded automatically from `data/CameraMask/` "
                "when you upload images from the test set."
            )
            with gr.Row():
                img1 = gr.Image(label="Image 1", type="filepath", height=150)
                img2 = gr.Image(label="Image 2", type="filepath", height=150)
                img3 = gr.Image(label="Image 3", type="filepath", height=150)
                img4 = gr.Image(label="Image 4", type="filepath", height=150)

            run_btn = gr.Button("Run Segmentation", variant="primary", size="lg")

            gr.Markdown("---\n### Results  *(Original · Ground Truth · Prediction)*")

            all_outputs = []
            for i in range(1, 5):
                gr.Markdown(f"**Sample {i}**")
                with gr.Row():
                    o_orig = gr.Image(
                        label=f"Sample {i} — Original",
                        height=200,
    
                    )
                    o_gt = gr.Image(
                        label=f"Sample {i} — Ground Truth",
                        height=200,
    
                    )
                    o_pred = gr.Image(
                        label=f"Sample {i} — Prediction",
                        height=200,
    
                    )
                all_outputs.extend([o_orig, o_gt, o_pred])

            run_btn.click(
                fn=infer_all,
                inputs=[img1, img2, img3, img4],
                outputs=all_outputs,
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft(),
                allowed_paths=[Q2_DIR, "data"])
