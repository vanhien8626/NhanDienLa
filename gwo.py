"""Grey Wolf Optimizer for mixed CNN hyperparameters (minimization)."""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout: float = 0.3
    optimizer: str = 'adamw'
    frozen_blocks: int = 0
    weight_decay: float = 0.0001


def decode(vector, max_frozen=2):
    x = np.asarray(vector, dtype=float)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('Expected six finite values.')
    x = np.clip(x, 0, 1)
    return Hyperparameters(
        learning_rate=round(float(10 ** (-4 + 2 * x[0])), 7),
        batch_size=(16, 32, 64)[min(2, int(x[1] * 3))],
        dropout=round(float(0.6 * x[2]), 3),
        optimizer=('adamw', 'sgd')[min(1, int(x[3] * 2))],
        frozen_blocks=min(max_frozen, int(x[4] * (max_frozen + 1))),
        weight_decay=round(float(10 ** (-6 + 4 * x[5])), 8),
    )


def optimize(objective, *, wolves=5, iterations=5, seed=42, max_frozen=2):
    if wolves < 3 or iterations < 1:
        raise ValueError('GWO requires at least 3 wolves and 1 iteration.')
    rng = np.random.default_rng(seed)
    positions = rng.random((wolves, 6))
    cache, history = {}, []
    best_config, best_score = None, float('inf')
    for iteration in range(iterations):
        scores = []
        for position in positions:
            config = decode(position, max_frozen)
            if config not in cache:
                score = float(objective(config))
                if not math.isfinite(score):
                    raise ValueError('Objective must return a finite score.')
                cache[config] = score
            score = cache[config]
            scores.append(score)
            if score < best_score:
                best_score, best_config = score, config
        leaders = positions[np.argsort(scores, kind='stable')[:3]].copy()
        history.append({'iteration': iteration + 1, 'best_fitness': best_score,
                        'evaluations': len(cache), 'best_config': asdict(best_config)})
        # a decreases from 2 towards 0. Updated wolves are evaluated next iteration.
        a = 2 * (1 - iteration / max(1, iterations - 1))
        candidates = []
        for leader in leaders:
            A = 2 * a * rng.random(positions.shape) - a
            C = 2 * rng.random(positions.shape)
            candidates.append(leader - A * np.abs(C * leader - positions))
        positions = np.clip(np.mean(candidates, axis=0), 0, 1)
    return best_config, history
