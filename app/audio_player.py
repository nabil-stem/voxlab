"""Lecteur audio simple basé sur sounddevice."""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None


class AudioPlayer:
    """Gère Play/Pause/Stop pour un waveform numpy."""

    def __init__(self) -> None:
        self._audio: np.ndarray | None = None
        self._sample_rate = 16000
        self._cursor = 0
        self._started_at = 0.0
        self._playing = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def has_audio(self) -> bool:
        return self._audio is not None and len(self._audio) > 0

    @property
    def duration_seconds(self) -> float:
        if self._audio is None:
            return 0.0
        return len(self._audio) / float(self._sample_rate)

    def set_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            self.stop()
            self._audio = np.asarray(audio, dtype=np.float32)
            self._sample_rate = int(sample_rate)
            self._cursor = 0

    def play(self, restart: bool = False) -> None:
        if sd is None:
            raise RuntimeError("sounddevice n'est pas installé.")
        with self._lock:
            if self._audio is None:
                raise RuntimeError("Aucun audio chargé.")
            if restart or self._cursor >= len(self._audio):
                self._cursor = 0
            sd.stop()
            sd.play(self._audio[self._cursor :], self._sample_rate, blocking=False)
            self._started_at = time.perf_counter()
            self._playing = True

    def pause(self) -> None:
        if sd is None:
            return
        with self._lock:
            if not self._playing:
                return
            elapsed = time.perf_counter() - self._started_at
            self._cursor += int(elapsed * self._sample_rate)
            self._cursor = min(self._cursor, len(self._audio) if self._audio is not None else 0)
            sd.stop()
            self._playing = False

    def stop(self) -> None:
        if sd is not None:
            sd.stop()
        self._cursor = 0
        self._started_at = 0.0
        self._playing = False

    def is_playing(self) -> bool:
        with self._lock:
            if not self._playing or self._audio is None:
                return False
            if self.position_seconds >= self.duration_seconds:
                self._playing = False
                self._cursor = len(self._audio)
                return False
            return True

    @property
    def position_seconds(self) -> float:
        if self._audio is None:
            return 0.0
        if self._playing:
            elapsed = time.perf_counter() - self._started_at
            frames = self._cursor + int(elapsed * self._sample_rate)
        else:
            frames = self._cursor
        frames = max(0, min(frames, len(self._audio)))
        return frames / float(self._sample_rate)
