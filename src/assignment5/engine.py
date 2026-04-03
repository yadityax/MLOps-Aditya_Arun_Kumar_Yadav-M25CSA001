from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from assignment5.utils import ensure_dir


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass(slots=True)
class TrainEpochResult:
    metrics: EpochMetrics
    gradient_norms: dict[str, float]


def extract_logits(outputs: object) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def _forward_model(model: torch.nn.Module, inputs: torch.Tensor) -> object:
    base_model = getattr(model, "base_model", model)
    model_type = getattr(getattr(base_model, "config", None), "model_type", None)
    if model_type == "vit":
        return model(pixel_values=inputs)
    return model(inputs)


def run_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    criterion = torch.nn.CrossEntropyLoss()
    is_training = optimizer is not None
    model.train(is_training)
    use_amp = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = _forward_model(model, inputs)
            logits = extract_logits(outputs)
            loss = criterion(logits, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


def train_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_prefixes: tuple[str, ...] = (),
) -> TrainEpochResult:
    criterion = torch.nn.CrossEntropyLoss()
    model.train(True)
    use_amp = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gradient_norms: dict[str, list[float]] = {prefix: [] for prefix in gradient_prefixes}

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = _forward_model(model, inputs)
            logits = extract_logits(outputs)
            loss = criterion(logits, targets)
        loss.backward()

        if gradient_prefixes:
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                for prefix in gradient_prefixes:
                    if prefix in name:
                        gradient_norms[prefix].append(parameter.grad.detach().norm().item())

        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    averaged_gradient_norms = {
        prefix: sum(values) / max(len(values), 1)
        for prefix, values in gradient_norms.items()
    }
    return TrainEpochResult(
        metrics=EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
        ),
        gradient_norms=averaged_gradient_norms,
    )


def evaluate(model: torch.nn.Module, loader: Iterable, device: torch.device) -> EpochMetrics:
    model.eval()
    with torch.no_grad():
        return run_epoch(model, loader, device)


def save_checkpoint(model: torch.nn.Module, output_dir: str | Path, name: str) -> Path:
    directory = ensure_dir(output_dir)
    path = directory / name
    torch.save(model.state_dict(), path)
    return path
