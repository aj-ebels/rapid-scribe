"""
Audio leveling utilities for microphone capture.

Provides a lightweight, stateful processing chain:
- Noise-floor tracking (for adaptive thresholds)
- Speech-aware AGC (bounded, smoothed gain)
- Gentle downward expansion below adaptive threshold
- Peak limiting
- Hangover-based activity detection to avoid chopping speech tails
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np


def _rms32(samples: np.ndarray) -> float:
    if samples is None or samples.size == 0:
        return 0.0
    buf = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(buf * buf)))


@dataclass
class AudioLevelerConfig:
    enabled: bool = True
    input_sensitivity: float = 0.8
    target_rms: float = 0.035
    min_gain_db: float = -12.0
    max_gain_db: float = 9.0
    attack: float = 0.25
    release: float = 0.06
    expander_enabled: bool = True
    expander_ratio: float = 3.0
    vad_floor: float = 0.0012
    vad_noise_multiplier: float = 2.8
    hangover_blocks: int = 6
    limiter_ceiling: float = 0.92
    noise_floor_alpha: float = 0.05


class AudioLeveler:
    def __init__(self, cfg: AudioLevelerConfig | None = None):
        self.cfg = cfg or AudioLevelerConfig()
        self._noise_floor = self.cfg.vad_floor
        self._gain = 1.0
        self._hangover = 0

    @property
    def noise_floor(self) -> float:
        return float(self._noise_floor)

    def current_threshold(self) -> float:
        return max(self.cfg.vad_floor, self._noise_floor * self.cfg.vad_noise_multiplier)

    def is_active(self, rms: float) -> bool:
        threshold = self.current_threshold()
        if rms >= threshold:
            self._hangover = self.cfg.hangover_blocks
            return True
        if self._hangover > 0:
            self._hangover -= 1
            return True
        return False

    def _update_noise_floor(self, rms_in: float):
        # Only learn floor from lower-energy regions so speech doesn't raise the baseline.
        upper = max(self.cfg.vad_floor * 6.0, self._noise_floor * 1.8)
        if rms_in <= upper:
            a = max(0.001, min(0.5, float(self.cfg.noise_floor_alpha)))
            self._noise_floor = (1.0 - a) * self._noise_floor + a * rms_in
            self._noise_floor = max(self.cfg.vad_floor * 0.5, self._noise_floor)

    def process(self, samples: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if samples is None or samples.size == 0:
            return np.asarray(samples, dtype=np.float32), {
                "rms_in": 0.0,
                "rms_out": 0.0,
                "noise_floor": self.noise_floor,
                "threshold": self.current_threshold(),
                "gain_db": 0.0,
                "clip_count": 0.0,
                "active": 0.0,
            }

        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        rms_in = _rms32(x)
        self._update_noise_floor(rms_in)
        active = self.is_active(rms_in)

        if not self.cfg.enabled:
            y = np.clip(x, -1.0, 1.0).astype(np.float32)
            return y, {
                "rms_in": rms_in,
                "rms_out": _rms32(y),
                "noise_floor": self.noise_floor,
                "threshold": self.current_threshold(),
                "gain_db": 0.0,
                "clip_count": 0.0,
                "active": 1.0 if active else 0.0,
            }

        eps = 1e-6
        target = max(self.cfg.vad_floor * 1.2, float(self.cfg.target_rms))
        desired_gain = (target / max(rms_in, eps)) * max(0.3, min(4.0, float(self.cfg.input_sensitivity)))
        if not active:
            # Do not chase noise floor upward when speech is absent.
            desired_gain = min(desired_gain, 1.0)
        min_g = math.pow(10.0, float(self.cfg.min_gain_db) / 20.0)
        max_g = math.pow(10.0, float(self.cfg.max_gain_db) / 20.0)
        desired_gain = max(min_g, min(max_g, desired_gain))

        smoothing = self.cfg.attack if desired_gain > self._gain else self.cfg.release
        smoothing = max(0.001, min(0.95, float(smoothing)))
        self._gain = (1.0 - smoothing) * self._gain + smoothing * desired_gain

        y = x * self._gain

        if self.cfg.expander_enabled:
            thr = self.current_threshold()
            if rms_in < thr:
                # Gentle attenuation below threshold instead of hard muting.
                rel = max(0.05, rms_in / max(thr, eps))
                attn = pow(rel, max(0.0, float(self.cfg.expander_ratio) - 1.0))
                y = y * attn
            if not active:
                # Extra suppression during non-speech windows reduces ASR hallucinations.
                y = y * 0.25

        ceiling = max(0.5, min(0.999, float(self.cfg.limiter_ceiling)))
        over = np.abs(y) > ceiling
        clip_count = int(np.count_nonzero(over))
        if clip_count:
            y = np.clip(y, -ceiling, ceiling)
        y = np.clip(y, -1.0, 1.0).astype(np.float32)

        gain_db = 20.0 * math.log10(max(self._gain, eps))
        return y, {
            "rms_in": rms_in,
            "rms_out": _rms32(y),
            "noise_floor": self.noise_floor,
            "threshold": self.current_threshold(),
            "gain_db": gain_db,
            "clip_count": float(clip_count),
            "active": 1.0 if active else 0.0,
        }
