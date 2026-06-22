"""Traitement audio pour simuler un style émotionnel.

Le TTS MMS-VITS produit une voix neutre. Ce module applique ensuite des
transformations DSP légères et explicables pour produire une prosodie plus
joyeuse, triste, calme ou en colère.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
from scipy import signal

from model.config import EMOTIONS, EmotionProfile


STYLE_VECTOR_ORDER = ("speed", "pitch", "gain_db", "brightness", "warmth", "drive")


def profile_to_vector(profile: EmotionProfile) -> np.ndarray:
    """Convertit un profil émotionnel en vecteur numérique."""

    data = asdict(profile)
    return np.array([float(data[name]) for name in STYLE_VECTOR_ORDER], dtype=np.float32)


def vector_to_profile(base_key: str, vector: Iterable[float]) -> EmotionProfile:
    """Crée un profil depuis un vecteur, en gardant label/couleur du profil."""

    base = EMOTIONS[base_key]
    values = list(vector)
    return EmotionProfile(
        key=base.key,
        label=base.label,
        speed=float(values[0]),
        pitch=float(values[1]),
        gain_db=float(values[2]),
        brightness=float(values[3]),
        warmth=float(values[4]),
        drive=float(values[5]),
        color=base.color,
    )


def normalize_audio(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    """Normalise le signal sans écraser totalement la dynamique."""

    arr = np.asarray(audio, dtype=np.float32)
    max_value = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_value <= 1e-8:
        return arr
    return (arr / max_value * peak).astype(np.float32)


def limit_peak(audio: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """Atténue uniquement si le signal dépasse le plafond.

    Contrairement à ``normalize_audio``, ce limiteur ne remonte jamais un
    signal faible. Cela permet à ``gain_db`` de réellement faire varier le
    volume entre émotions (colère plus forte, tristesse plus douce) tout en
    évitant l'écrêtage.
    """

    arr = np.asarray(audio, dtype=np.float32)
    max_value = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_value > ceiling:
        return (arr / max_value * ceiling).astype(np.float32)
    return arr


def _safe_filter(btype: str, cutoff: float, sample_rate: int, order: int = 2):
    nyquist = sample_rate / 2.0
    safe_cutoff = max(40.0, min(cutoff, nyquist - 100.0))
    return signal.butter(order, safe_cutoff / nyquist, btype=btype)


def _resample(audio: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 4 or len(audio) <= 4:
        return audio.astype(np.float32)
    return signal.resample(audio, target_len).astype(np.float32)


def _apply_speed_and_pitch(audio: np.ndarray, profile: EmotionProfile) -> np.ndarray:
    """Applique une transformation simple de vitesse et de hauteur.

    Cette approche reste volontairement légère. Pour un vrai pitch shifting
    indépendant du tempo, il faudrait un phase vocoder ou un modèle prosodique
    entraîné. Ici, on privilégie une solution locale, lisible et robuste.
    """

    arr = audio.astype(np.float32)

    # Le pitch agit légèrement sur la longueur du signal. Une valeur supérieure
    # à 1 donne une voix plus aiguë et plus vive.
    pitch = max(0.70, min(1.35, profile.pitch))
    if abs(pitch - 1.0) > 0.01:
        arr = _resample(arr, int(len(arr) / pitch))

    # La vitesse contrôle la durée finale de manière prévisible.
    speed = max(0.60, min(1.45, profile.speed))
    if abs(speed - 1.0) > 0.01:
        arr = _resample(arr, int(len(arr) / speed))

    return arr


def _apply_tone(audio: np.ndarray, sample_rate: int, profile: EmotionProfile) -> np.ndarray:
    """Modifie la couleur de la voix: brillance, chaleur et saturation."""

    arr = audio.astype(np.float32)

    if abs(profile.brightness) > 0.01 and len(arr) > 16:
        low_b, low_a = _safe_filter("lowpass", 2800.0, sample_rate)
        low = signal.filtfilt(low_b, low_a, arr).astype(np.float32)
        high = arr - low
        if profile.brightness > 0:
            arr = arr + high * profile.brightness
        else:
            arr = arr * (1.0 + profile.brightness * 0.25) + low * abs(profile.brightness)

    if profile.warmth > 0.01 and len(arr) > 16:
        low_b, low_a = _safe_filter("lowpass", 850.0, sample_rate)
        low = signal.filtfilt(low_b, low_a, arr).astype(np.float32)
        arr = arr + low * min(profile.warmth, 0.60) * 0.35

    if profile.drive > 0.01:
        drive = min(profile.drive, 0.45)
        arr = np.tanh(arr * (1.0 + drive * 4.0)).astype(np.float32)

    return arr


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    factor = 10.0 ** (gain_db / 20.0)
    return (audio.astype(np.float32) * factor).astype(np.float32)


def apply_emotion(audio: np.ndarray, sample_rate: int, profile: EmotionProfile) -> np.ndarray:
    """Applique un profil émotionnel complet à un waveform mono."""

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # On normalise vers un niveau de référence avec de la marge AVANT le gain,
    # de sorte que gain_db modifie réellement le volume final. Un simple
    # limiteur en sortie évite l'écrêtage sans annuler ces différences.
    arr = normalize_audio(audio, peak=0.72)
    arr = _apply_speed_and_pitch(arr, profile)
    arr = _apply_tone(arr, sample_rate, profile)
    arr = _apply_gain(arr, profile.gain_db)
    return limit_peak(arr, ceiling=0.97)


def apply_emotion_by_key(audio: np.ndarray, sample_rate: int, emotion_key: str) -> np.ndarray:
    """Raccourci utilisé par les scripts et tests."""

    profile = EMOTIONS.get(emotion_key, EMOTIONS["neutral"])
    return apply_emotion(audio, sample_rate, profile)
