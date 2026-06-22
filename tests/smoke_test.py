"""Smoke tests pour VoxLab.

Tests rapides:
    python tests/smoke_test.py

Tests rapides + génération TTS réelle:
    python tests/smoke_test.py --with-tts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from tkinter import messagebox

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.audio_player import AudioPlayer
from app.gui import VoxLabApp
from model.config import EMOTIONS
from model.emotion import apply_emotion
from model.style_model import predict_profile_with_checkpoint
from model.tts_engine import TTSEngine


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_emotion_processing() -> None:
    sample_rate = 16000
    t = np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    peaks = {}
    for key, profile in EMOTIONS.items():
        styled = apply_emotion(audio, sample_rate, profile)
        check(styled.dtype == np.float32, f"{key}: dtype inattendu")
        check(len(styled) > 1000, f"{key}: audio trop court")
        peak = float(np.max(np.abs(styled)))
        check(peak <= 0.9701, f"{key}: écrêtage (peak={peak:.3f})")
        peaks[key] = peak

    # gain_db doit réellement influencer le volume: la colère (gain positif)
    # doit sortir plus fort que la tristesse (gain négatif).
    check(
        peaks["anger"] > peaks["sadness"] + 0.05,
        f"gain_db sans effet: anger={peaks['anger']:.3f} sadness={peaks['sadness']:.3f}",
    )


def test_style_profile() -> None:
    profile = predict_profile_with_checkpoint("joy")
    check(profile.key == "joy", "profil émotionnel incorrect")
    check(0.6 <= profile.speed <= 1.45, "speed hors limites")
    check(0.7 <= profile.pitch <= 1.35, "pitch hors limites")


def test_gui_creation() -> None:
    messagebox.showerror = lambda *args, **kwargs: None
    app = VoxLabApp()
    check(bool(app.home_frame.grid_info()), "page d'accueil non affichée au démarrage")
    app._show_studio()
    check(bool(app.studio_frame.grid_info()), "studio non affiché")
    app._show_home()
    check(bool(app.home_frame.grid_info()), "retour accueil impossible")
    app._generation_failed(ValueError("test"))
    app.after(150, app.destroy)
    app.mainloop()


def test_audio_player() -> None:
    player = AudioPlayer()
    if not player.available:
        print("SKIP audio player: sounddevice indisponible")
        return

    player.set_audio(np.zeros(1600, dtype=np.float32), 16000)
    player.play(restart=True)
    time.sleep(0.04)
    check(player.position_seconds > 0, "position de lecture non mise à jour")
    player.pause()
    player.stop()


def test_real_tts() -> None:
    engine = TTSEngine()
    check(engine.available, "moteur TTS indisponible")
    result = engine.synthesize(
        "Bonjour, test automatique de VoxLab.",
        emotion_key="neutral",
        language="fr",
        use_style_model=True,
    )
    check(result.audio_path.exists(), "WAV non créé")
    check(result.waveform_path.exists(), "waveform PNG non créée")
    check(result.duration_seconds > 0.2, "durée audio trop courte")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke tests VoxLab.")
    parser.add_argument("--with-tts", action="store_true", help="inclut une synthèse TTS réelle")
    args = parser.parse_args()

    tests = [
        ("emotion-processing", test_emotion_processing),
        ("style-profile", test_style_profile),
        ("gui-creation", test_gui_creation),
        ("audio-player", test_audio_player),
    ]
    if args.with_tts:
        tests.append(("real-tts", test_real_tts))

    for name, func in tests:
        func()
        print(f"OK {name}")


if __name__ == "__main__":
    main()
