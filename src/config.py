from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    target_cols: List[str] = field(default_factory=lambda: [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ])
    in_channels: int = 5
    image_size: int = 256
    n_splits: int = 5
    fold: int = 0
    epochs: int = 5
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-2
    eta_min: float = 1e-6
    gold_sample_weight: float = 1.0  # Upgrade in v2.0 (e.g., 2.5x weight on gold studies)
    seed: int = 42
