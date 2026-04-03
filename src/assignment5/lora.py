from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TrainableParameterReport:
    total_parameters: int
    trainable_parameters: int

    @property
    def trainable_ratio(self) -> float:
        return self.trainable_parameters / max(self.total_parameters, 1)


def freeze_backbone_except_head(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier") or name.startswith("fc") or name.startswith("head")


def count_parameters(model: torch.nn.Module) -> TrainableParameterReport:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return TrainableParameterReport(total_parameters=total, trainable_parameters=trainable)


def set_trainable_module_names(model: torch.nn.Module, names: tuple[str, ...]) -> None:
    for module_name, parameter in model.named_parameters():
        parameter.requires_grad = any(module_name.startswith(name) for name in names)
