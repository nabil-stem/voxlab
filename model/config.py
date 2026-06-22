"""Configuration centrale du projet VoxLab.

Les paramètres restent lisibles pour pouvoir expliquer facilement comment
chaque émotion influence le signal audio généré.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUT_DIR = ROOT_DIR / "outputs"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
WAVEFORM_OUTPUT_DIR = OUTPUT_DIR / "waveforms"
CHECKPOINT_DIR = ROOT_DIR / "model" / "checkpoints"
STYLE_CHECKPOINT_PATH = CHECKPOINT_DIR / "emotion_style.pt"


HF_TTS_MODELS = {
    "fr": "facebook/mms-tts-fra",
    "en": "facebook/mms-tts-eng",
}

LANGUAGE_LABELS = {
    "fr": "Français",
    "en": "English",
}


@dataclass(frozen=True)
class EmotionProfile:
    """Paramètres de style appliqués après la synthèse MMS-TTS.

    speed:
        Plus la valeur est grande, plus la voix est rapide.
    pitch:
        Modifie la hauteur de façon légère par rééchantillonnage.
    gain_db:
        Volume ajouté ou réduit en décibels.
    brightness:
        Valeur positive = voix plus brillante, négative = voix plus feutrée.
    warmth:
        Renforce légèrement les graves et le corps de la voix.
    drive:
        Saturation douce. Utile pour les émotions intenses comme la colère.
    """

    key: str
    label: str
    speed: float
    pitch: float
    gain_db: float
    brightness: float
    warmth: float
    drive: float
    color: str


EMOTIONS = {
    "neutral": EmotionProfile(
        key="neutral",
        label="Neutre",
        speed=1.00,
        pitch=1.00,
        gain_db=0.0,
        brightness=0.00,
        warmth=0.10,
        drive=0.00,
        color="#7dd3fc",
    ),
    "joy": EmotionProfile(
        key="joy",
        label="Joie",
        speed=1.12,
        pitch=1.08,
        gain_db=1.5,
        brightness=0.45,
        warmth=0.05,
        drive=0.04,
        color="#facc15",
    ),
    "sadness": EmotionProfile(
        key="sadness",
        label="Tristesse",
        speed=0.82,
        pitch=0.93,
        gain_db=-2.0,
        brightness=-0.35,
        warmth=0.35,
        drive=0.00,
        color="#60a5fa",
    ),
    "anger": EmotionProfile(
        key="anger",
        label="Colère",
        speed=1.18,
        pitch=0.97,
        gain_db=3.0,
        brightness=0.20,
        warmth=0.15,
        drive=0.22,
        color="#fb7185",
    ),
    "calm": EmotionProfile(
        key="calm",
        label="Calme",
        speed=0.90,
        pitch=0.98,
        gain_db=-1.5,
        brightness=-0.20,
        warmth=0.45,
        drive=0.00,
        color="#5eead4",
    ),
}

EMOTION_LABEL_TO_KEY = {profile.label: key for key, profile in EMOTIONS.items()}


def ensure_project_dirs() -> None:
    """Crée les dossiers nécessaires au runtime."""

    for path in (AUDIO_OUTPUT_DIR, WAVEFORM_OUTPUT_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)
