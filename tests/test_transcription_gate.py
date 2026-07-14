"""Unit tests for adaptive transcription RMS gating (long-call gate creep fix)."""
from app.transcription import (
    initial_transcription_noise_floor,
    update_transcription_gate,
)


def test_initial_noise_floor_tracks_min_rms():
    assert initial_transcription_noise_floor(0.01) == 0.01
    assert initial_transcription_noise_floor(0.001) == 0.0025


def test_quiet_chunks_raise_floor_but_cap_keeps_speech_passing():
    """Simulate a long call: speech-level RMS must not close the gate."""
    min_rms = 0.01
    floor = initial_transcription_noise_floor(min_rms)

    # Warm-up on near-silence (legitimate learning).
    for rms in (0.002, 0.003, 0.004, 0.005):
        _, floor = update_transcription_gate(min_rms, floor, rms, adaptive_gate=True)

    # Many minutes of meeting speech at typical loopback levels (~0.03–0.05 RMS).
    for _ in range(200):
        effective, floor = update_transcription_gate(min_rms, floor, 0.042, adaptive_gate=True)
        assert effective < 0.042, f"gate closed on speech-level audio: {effective}"
        assert 0.042 >= effective


def test_speech_rms_does_not_inflate_noise_floor():
    min_rms = 0.01
    floor = initial_transcription_noise_floor(min_rms)
    _, floor_after = update_transcription_gate(min_rms, floor, 0.045, adaptive_gate=True)
    assert floor_after == floor


def test_adaptive_disabled_uses_base_threshold_only():
    min_rms = 0.01
    floor = 0.03
    effective, new_floor = update_transcription_gate(min_rms, floor, 0.045, adaptive_gate=False)
    assert effective == min_rms
    assert new_floor == floor


def test_gate_reset_restores_baseline():
    min_rms = 0.01
    floor = initial_transcription_noise_floor(min_rms)
    for rms in (0.004, 0.005, 0.006, 0.007, 0.008):
        _, floor = update_transcription_gate(min_rms, floor, rms, adaptive_gate=True)
    assert floor > 0.005
    reset_floor = initial_transcription_noise_floor(min_rms)
    assert reset_floor == min_rms
