import torch
from torch import nn

from missingness.models.predictor import ProbabilityEnsemble


def test_predictor_averages_probabilities_before_converting_to_logits():
    first, second = nn.Linear(1, 1), nn.Linear(1, 1)
    with torch.no_grad():
        first.bias.fill_(-3)
        second.bias.fill_(2)
    ensemble = ProbabilityEnsemble([first, second], [0.25, 0.75])
    x = torch.zeros(2, 3, 1)
    expected = 0.25 * first(x).sigmoid() + 0.75 * second(x).sigmoid()
    torch.testing.assert_close(ensemble(x).sigmoid(), expected)
