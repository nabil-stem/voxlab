"""Génère un dataset jouet pour VoxLab.

Le script crée des fichiers WAV dans dataset/audio/ et un fichier metadata.csv.
Il essaie d'abord pyttsx3 pour générer une voix locale. Si pyttsx3 ou le moteur
vocal système échoue, il produit un signal synthétique de secours.

Lancement:
    python dataset/create_toy_dataset.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.config import DATASET_DIR, EMOTIONS, ensure_project_dirs
from model.emotion import apply_emotion

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None


SAMPLE_RATE = 22050
TEXTS = [
    "Bonjour, ceci est un exemple de synthèse vocale expressive.",
    "Je suis heureux de vous présenter ce projet Python.",
    "La météo est sombre aujourd'hui, mais la voix reste claire.",
    "Attention, le système doit répondre rapidement et fortement.",
    "Prenons un moment pour parler calmement et simplement.",
    "VoxLab génère des exemples audio pour entraîner un petit modèle.",
]


def _speech_like_signal(text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Produit un signal de secours proche d'une voix, sans être intelligible."""

    clean = text.strip() or "texte"
    rng = np.random.default_rng(abs(hash(clean)) % (2**32))
    chunks: list[np.ndarray] = []

    for index, char in enumerate(clean[:140]):
        if char.isspace():
            chunks.append(np.zeros(int(sample_rate * 0.035), dtype=np.float32))
            continue

        duration = 0.045 + (ord(char) % 5) * 0.006
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        base_freq = 145 + (ord(char.lower()) % 32) * 5
        vibrato = 1.0 + 0.015 * np.sin(2 * np.pi * 5.2 * t + index * 0.1)
        tone = (
            0.55 * np.sin(2 * np.pi * base_freq * vibrato * t)
            + 0.24 * np.sin(2 * np.pi * base_freq * 2.0 * t)
            + 0.10 * np.sin(2 * np.pi * base_freq * 3.0 * t)
        )
        envelope = np.sin(np.linspace(0, np.pi, len(t))) ** 0.75
        noise = rng.normal(0.0, 0.012, size=len(t))
        chunks.append(((tone + noise) * envelope).astype(np.float32))

    if not chunks:
        return np.zeros(sample_rate // 2, dtype=np.float32)
    audio = np.concatenate(chunks)
    peak = np.max(np.abs(audio)) or 1.0
    return (audio / peak * 0.75).astype(np.float32)


def _try_pyttsx3(text: str, out_path: Path) -> bool:
    """Retourne True si pyttsx3 a bien écrit un fichier audio lisible."""

    if pyttsx3 is None:
        return False

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 0.95)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception:
        return False


def _base_audio_for_text(text: str, tmp_dir: Path) -> tuple[np.ndarray, int]:
    base_path = tmp_dir / "base.wav"
    if _try_pyttsx3(text, base_path):
        try:
            audio, sample_rate = sf.read(base_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(sample_rate)
        except Exception:
            pass

    return _speech_like_signal(text), SAMPLE_RATE


def generate_dataset() -> Path:
    ensure_project_dirs()
    audio_dir = DATASET_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = DATASET_DIR / "metadata.csv"

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        item_id = 0
        for text_index, text in enumerate(TEXTS):
            base_audio, sample_rate = _base_audio_for_text(text, tmp_dir)
            for emotion_key, profile in EMOTIONS.items():
                item_id += 1
                styled = apply_emotion(base_audio, sample_rate, profile)
                filename = f"sample_{text_index:02d}_{emotion_key}.wav"
                file_path = audio_dir / filename
                sf.write(file_path, styled, sample_rate)
                rows.append(
                    {
                        "id": item_id,
                        "file_path": str(file_path.as_posix()),
                        "text": text,
                        "emotion": emotion_key,
                        "language": "fr",
                        "duration_seconds": round(len(styled) / sample_rate, 3),
                        "sample_rate": sample_rate,
                        "speed": profile.speed,
                        "pitch": profile.pitch,
                        "gain_db": profile.gain_db,
                        "brightness": profile.brightness,
                        "warmth": profile.warmth,
                        "drive": profile.drive,
                    }
                )

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


def main() -> None:
    metadata = generate_dataset()
    print(f"Dataset généré: {metadata}")
    print("Vous pouvez ensuite lancer: python -m model.train_style_model")


if __name__ == "__main__":
    main()
