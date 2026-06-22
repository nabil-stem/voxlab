"""Petit réseau de neurones pour prédire les paramètres émotionnels.

Le TTS principal reste le modèle pré-entraîné MMS-TTS. Ce réseau sert à
illustrer la partie entraînement/fine-tuning légère du projet: il apprend à
mapper une émotion vers un vecteur de style DSP.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from model.config import EMOTIONS, STYLE_CHECKPOINT_PATH, EmotionProfile
from model.emotion import STYLE_VECTOR_ORDER, profile_to_vector, vector_to_profile

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - utile si torch n'est pas installé
    torch = None
    nn = None


EMOTION_KEYS = list(EMOTIONS.keys())


class EmotionStyleNet(nn.Module):
    """MLP compact: émotion one-hot -> paramètres de style."""

    def __init__(self, emotion_count: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emotion_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _one_hot(emotion_key: str) -> np.ndarray:
    vector = np.zeros(len(EMOTION_KEYS), dtype=np.float32)
    vector[EMOTION_KEYS.index(emotion_key)] = 1.0
    return vector


def _emotion_from_metadata_value(value: str) -> str:
    clean = value.strip().lower()
    label_map = {profile.label.lower(): key for key, profile in EMOTIONS.items()}
    return label_map.get(clean, clean if clean in EMOTIONS else "neutral")


def _read_metadata_emotions(metadata_path: Path | None) -> list[str]:
    if metadata_path is None or not metadata_path.exists():
        return []

    emotions: list[str] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emotions.append(_emotion_from_metadata_value(row.get("emotion", "neutral")))
    return emotions


def build_training_arrays(metadata_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Construit un dataset minuscule à partir des profils et métadonnées."""

    keys = _read_metadata_emotions(metadata_path)
    if not keys:
        keys = EMOTION_KEYS * 16

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    rng = np.random.default_rng(42)
    for key in keys:
        base = profile_to_vector(EMOTIONS.get(key, EMOTIONS["neutral"]))
        for _ in range(3):
            # Petite variation pour éviter que le modèle mémorise un seul point.
            noise = rng.normal(0.0, [0.015, 0.010, 0.08, 0.015, 0.015, 0.008])
            inputs.append(_one_hot(key))
            targets.append((base + noise).astype(np.float32))

    return np.vstack(inputs).astype(np.float32), np.vstack(targets).astype(np.float32)


def train_style_model(
    metadata_path: Path | None = None,
    checkpoint_path: Path = STYLE_CHECKPOINT_PATH,
    epochs: int = 450,
    learning_rate: float = 0.015,
) -> Path:
    """Entraîne le petit modèle et sauvegarde un checkpoint."""

    if torch is None:
        raise RuntimeError("PyTorch n'est pas installé. Lancez: pip install torch")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    x_np, y_np = build_training_arrays(metadata_path)

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    model = EmotionStyleNet(len(EMOTION_KEYS), len(STYLE_VECTOR_ORDER))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "emotion_keys": EMOTION_KEYS,
            "style_vector_order": STYLE_VECTOR_ORDER,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _clamp_style_vector(vector: Iterable[float]) -> np.ndarray:
    arr = np.array(list(vector), dtype=np.float32)
    arr[0] = np.clip(arr[0], 0.60, 1.45)   # speed
    arr[1] = np.clip(arr[1], 0.70, 1.35)   # pitch
    arr[2] = np.clip(arr[2], -6.0, 6.0)    # gain_db
    arr[3] = np.clip(arr[3], -0.70, 0.70)  # brightness
    arr[4] = np.clip(arr[4], 0.00, 0.70)   # warmth
    arr[5] = np.clip(arr[5], 0.00, 0.45)   # drive
    return arr


def predict_profile_with_checkpoint(
    emotion_key: str,
    checkpoint_path: Path = STYLE_CHECKPOINT_PATH,
) -> EmotionProfile:
    """Retourne un profil appris, ou le profil par défaut si absent."""

    if torch is None or not checkpoint_path.exists():
        return EMOTIONS.get(emotion_key, EMOTIONS["neutral"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    keys = list(checkpoint.get("emotion_keys", EMOTION_KEYS))
    if emotion_key not in keys:
        return EMOTIONS["neutral"]

    model = EmotionStyleNet(len(keys), len(STYLE_VECTOR_ORDER))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x = np.zeros(len(keys), dtype=np.float32)
    x[keys.index(emotion_key)] = 1.0
    with torch.no_grad():
        vector = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()

    vector = _clamp_style_vector(vector)
    return vector_to_profile(emotion_key, vector)
