"""Moteur TTS basé sur Facebook MMS-TTS/VITS via Hugging Face."""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from model.config import (
    AUDIO_OUTPUT_DIR,
    EMOTIONS,
    HF_TTS_MODELS,
    WAVEFORM_OUTPUT_DIR,
    ensure_project_dirs,
)
from model.emotion import apply_emotion
from model.style_model import predict_profile_with_checkpoint

try:
    import torch
    from transformers import AutoTokenizer, VitsModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except ImportError:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    VitsModel = None

try:
    from huggingface_hub.utils import logging as hub_logging

    hub_logging.set_verbosity_error()
except ImportError:  # pragma: no cover
    pass

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated.*",
    category=FutureWarning,
)


@dataclass
class SynthesisResult:
    """Résultat complet d'une génération audio."""

    audio: np.ndarray
    sample_rate: int
    audio_path: Path
    waveform_path: Path
    text: str
    emotion_key: str
    emotion_label: str
    language: str
    duration_seconds: float


class TTSEngine:
    """Charge les modèles MMS-TTS et synthétise un texte en audio."""

    def __init__(self) -> None:
        ensure_project_dirs()
        self._models: dict[str, tuple[object, object]] = {}
        self._device = "cpu"
        if torch is not None and torch.cuda.is_available():
            self._device = "cuda"

    @property
    def device(self) -> str:
        return self._device

    @property
    def available(self) -> bool:
        return torch is not None and AutoTokenizer is not None and VitsModel is not None

    def is_loaded(self, language: str) -> bool:
        return language in self._models

    def load_model(self, language: str = "fr") -> None:
        """Télécharge au premier lancement puis garde le modèle en mémoire."""

        if not self.available:
            raise RuntimeError(
                "Dépendances manquantes. Installez: pip install -r requirements.txt"
            )
        if language not in HF_TTS_MODELS:
            raise ValueError(f"Langue inconnue: {language}")
        if language in self._models:
            return

        model_id = HF_TTS_MODELS[language]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        model.to(self._device)
        model.eval()
        self._models[language] = (model, tokenizer)

    def synthesize(
        self,
        text: str,
        emotion_key: str = "neutral",
        language: str = "fr",
        use_style_model: bool = True,
    ) -> SynthesisResult:
        """Génère un fichier WAV et son image waveform."""

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Le texte est vide.")

        self.load_model(language)
        model, tokenizer = self._models[language]

        inputs = tokenizer(clean_text, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        audio = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32)
        sample_rate = int(model.config.sampling_rate)

        if use_style_model:
            profile = predict_profile_with_checkpoint(emotion_key)
        else:
            profile = EMOTIONS.get(emotion_key, EMOTIONS["neutral"])

        styled_audio = apply_emotion(audio, sample_rate, profile)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        slug = _slugify(clean_text[:40]) or "tts"
        stem = f"{timestamp}_{language}_{emotion_key}_{slug}"
        audio_path = AUDIO_OUTPUT_DIR / f"{stem}.wav"
        waveform_path = WAVEFORM_OUTPUT_DIR / f"{stem}.png"

        sf.write(audio_path, styled_audio, sample_rate)
        save_waveform_png(styled_audio, sample_rate, waveform_path, profile.label, profile.color)

        return SynthesisResult(
            audio=styled_audio,
            sample_rate=sample_rate,
            audio_path=audio_path,
            waveform_path=waveform_path,
            text=clean_text,
            emotion_key=emotion_key,
            emotion_label=profile.label,
            language=language,
            duration_seconds=float(len(styled_audio) / sample_rate),
        )


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value[:50]


def save_waveform_png(
    audio: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    color: str,
) -> Path:
    """Sauvegarde une image PNG de la forme d'onde."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    max_points = 7000
    if len(arr) > max_points:
        indices = np.linspace(0, len(arr) - 1, max_points).astype(np.int64)
        plot_audio = arr[indices]
        time_axis = indices / sample_rate
    else:
        plot_audio = arr
        time_axis = np.arange(len(arr)) / sample_rate

    fig = Figure(figsize=(9.5, 2.2), dpi=130, facecolor="#07111f")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#07111f")
    ax.plot(time_axis, plot_audio, color=color, linewidth=0.85)
    ax.axhline(0, color="#1f2937", linewidth=0.7)
    ax.set_title(f"Waveform - {title}", color="#e5e7eb", fontsize=10, pad=8)
    ax.set_xlabel("Temps (s)", color="#9ca3af", fontsize=8)
    ax.set_ylabel("Amplitude", color="#9ca3af", fontsize=8)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    fig.tight_layout(pad=1.0)
    fig.savefig(path, facecolor=fig.get_facecolor())
    return path
