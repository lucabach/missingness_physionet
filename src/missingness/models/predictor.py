"""Load a validation-selected predictor for inference on the usual normalized batches."""

import json
from pathlib import Path

import torch
import yaml
from torch import nn

from missingness.data.preprocessing import Normalizer
from missingness.models import build_model
from missingness.paths import OUTPUT_DIR


class ProbabilityEnsemble(nn.Module):
    def __init__(self, members: list[nn.Module], weights: list[float]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, batch: dict) -> torch.Tensor:
        """Return one logit per hour, compatible with experiment.predict."""
        probabilities = torch.stack([model(batch).sigmoid() for model in self.members], dim=-1)
        return torch.logit(probabilities @ self.weights, eps=torch.finfo(probabilities.dtype).eps)


def load_predictor(report: Path, outputs: Path = OUTPUT_DIR) -> tuple[nn.Module, Normalizer]:
    """Reconstruct the selected model and training normalizer from its aggregate JSON report."""
    weights = json.loads(report.read_text())["weights"]
    members = []
    norm_data = None
    for name in weights:
        run = outputs / name
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        stats = json.loads((run / "normalizer.json").read_text())
        if norm_data is not None and stats != norm_data:
            raise ValueError("ensemble checkpoints must use the same training normalizer")
        norm_data = stats
        member = build_model(cfg["model"], len(stats["mean"]), len(stats["static_mean"]), "all")
        member.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
        members.append(member)
    return ProbabilityEnsemble(members, list(weights.values())).eval(), Normalizer.from_dict(
        norm_data
    )
