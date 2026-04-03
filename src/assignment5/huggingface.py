from __future__ import annotations

from pathlib import Path

import torch


def save_and_push_to_huggingface(
    model: torch.nn.Module,
    checkpoint_path: Path,
    repo_id: str,
    hf_token: str | None = None,
    private: bool = False,
) -> str:
    """Save model checkpoint and push to Hugging Face Hub."""
    try:
        from transformers import ViTForImageClassification
        from huggingface_hub import create_repo, upload_folder
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face libraries are required. Install transformers and huggingface_hub."
        ) from exc

    model.eval()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    repo_url = None
    try:
        repo_url = create_repo(repo_id, private=private, exist_ok=True, token=hf_token)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: Could not create repo {repo_id}: {exc}")
        return f"https://huggingface.co/{repo_id}"

    local_dir = checkpoint_path.parent / "hf_export"
    local_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, torch.nn.Module) and hasattr(model, "save_pretrained"):
        model.save_pretrained(local_dir)
    else:
        torch.save(model.state_dict(), local_dir / "pytorch_model.bin")

    try:
        upload_folder(repo_id=repo_id, folder_path=str(local_dir), token=hf_token)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: Could not upload to repo {repo_id}: {exc}")

    return f"https://huggingface.co/{repo_id}"
