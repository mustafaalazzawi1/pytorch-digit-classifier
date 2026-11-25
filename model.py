"""A small PyTorch image classifier with an inspectable JSON export."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, inputs):
        return self.layers(inputs)


def normalize(pixels):
    values = np.asarray(pixels, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 64 or len(values) == 0:
        raise ValueError('Expected one or more flattened 8×8 images (64 pixels each).')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 16).any():
        raise ValueError('Pixels must be finite numbers between 0 and 16.')
    return torch.from_numpy(values / 16.0)


def probabilities(model, pixels):
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(normalize(pixels)), dim=1).numpy()


def export_model(model, path):
    data = {'format_version': 1, 'architecture': '64-ReLU64-10', 'pixel_max': 16,
            'state': {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}}
    Path(path).write_text(json.dumps(data, allow_nan=False) + '\n')


def load_model(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('format_version') != 1 or data.get('architecture') != '64-ReLU64-10' or data.get('pixel_max') != 16:
        raise ValueError('Unsupported model format or preprocessing.')
    model = DigitClassifier()
    if not isinstance(data.get('state'), dict) or set(data['state']) != set(model.state_dict()):
        raise ValueError('Missing or unexpected model tensors.')
    state = {}
    for name, expected in model.state_dict().items():
        value = torch.tensor(data['state'][name], dtype=torch.float32)
        if value.shape != expected.shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid model tensor: {name}')
        state[name] = value
    model.load_state_dict(state)
    model.eval()
    return model
