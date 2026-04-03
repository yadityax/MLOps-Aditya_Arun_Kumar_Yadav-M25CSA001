from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingConfig:
    data_root: str = "data"
    output_dir: str = "outputs"
    batch_size: int = 64
    # Default to 0 for Docker stability; increase only when container shm is large enough.
    num_workers: int = 0
    image_size: int = 224
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 10
    seed: int = 42
    device: str = "cuda"
    wandb_project: str = "assignment5"
    wandb_entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    pretrained_model_name: str = "WinKawaks/vit-small-patch16-224"


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 8
    alpha: int = 8
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "key", "value")


@dataclass(slots=True)
class AttackConfig:
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 10
    random_start: bool = True
