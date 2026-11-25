"""The original UCI writer-separated train/test split, with train-only validation."""
import hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent / 'data'
SHA256 = {
 'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
 'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8',
}


def load_data(directory=DATA):
    result = []
    for name, count in [('optdigits.tra', 3823), ('optdigits.tes', 1797)]:
        path = Path(directory) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256[name]:
            raise ValueError(f'Dataset checksum mismatch: {name}')
        table = np.loadtxt(path, delimiter=',', dtype=np.int64)
        if table.shape != (count, 65):
            raise ValueError(f'Unexpected dataset shape: {name}')
        result.append((table[:, :64], table[:, 64]))
    return result[0], result[1]


def split_training(labels, seed=2026):
    """Split only official training rows, retaining disjoint indices for auditing."""
    indices = np.arange(len(labels))
    train, validation = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
    return np.sort(train), np.sort(validation)
