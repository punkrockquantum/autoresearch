"""Recording from the glasses when they're paired as a Bluetooth headset.

The glasses expose a normal HFP/A2DP microphone, so any machine they're paired
with can capture from them — this is the one route that needs no Meta API, no
messaging app and no webhook, and it is the only one that can hold a long
prompt. Speech is endpointed with a plain RMS gate: talking starts the clip,
about a second of quiet ends it.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass


class MicError(RuntimeError):
    pass


@dataclass
class MicSettings:
    samplerate: int = 16000
    channels: int = 1
    device: int | str | None = None
    block_seconds: float = 0.05
    #: RMS of int16 samples normalised to 0..1. Bluetooth mics run quiet; lower
    #: this if clips never start, raise it if room noise triggers recording.
    silence_threshold: float = 0.012
    #: Quiet time that ends a clip.
    silence_seconds: float = 1.1
    max_seconds: float = 45.0
    min_seconds: float = 0.5
    #: How long to wait for speech to begin (None waits forever).
    start_timeout: float | None = None


def _require_sounddevice():
    try:
        import numpy  # noqa: F401
        import sounddevice
    except ImportError as exc:  # pragma: no cover - optional extra
        raise MicError(
            "microphone capture needs the 'mic' extra: pip install 'voicebridge[mic]'"
        ) from exc
    return sounddevice


def list_input_devices() -> list[dict[str, object]]:
    sounddevice = _require_sounddevice()
    devices = []
    for index, info in enumerate(sounddevice.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": index,
                    "name": info.get("name", "?"),
                    "channels": info.get("max_input_channels"),
                    "default_samplerate": info.get("default_samplerate"),
                }
            )
    return devices


def record_until_silence(settings: MicSettings | None = None) -> bytes:
    """Block until a phrase has been spoken, then return it as WAV bytes."""
    settings = settings or MicSettings()
    sounddevice = _require_sounddevice()
    import numpy as np

    block_frames = max(1, int(settings.samplerate * settings.block_seconds))
    silence_blocks = max(1, int(settings.silence_seconds / settings.block_seconds))
    max_blocks = int(settings.max_seconds / settings.block_seconds)
    start_blocks = (
        int(settings.start_timeout / settings.block_seconds) if settings.start_timeout else None
    )

    collected: list["np.ndarray"] = []
    quiet_run = 0
    waited = 0
    started = False

    try:
        stream = sounddevice.InputStream(
            samplerate=settings.samplerate,
            channels=settings.channels,
            dtype="int16",
            blocksize=block_frames,
            device=settings.device,
        )
    except Exception as exc:  # noqa: BLE001 - PortAudio raises many types
        raise MicError(f"could not open the input device: {exc}") from None

    with stream:
        while True:
            block, overflowed = stream.read(block_frames)
            if overflowed:
                continue
            level = float(np.sqrt(np.mean(np.square(block.astype(np.float32) / 32768.0))))
            if not started:
                if level >= settings.silence_threshold:
                    started = True
                    collected.append(block.copy())
                    continue
                waited += 1
                if start_blocks is not None and waited >= start_blocks:
                    return b""
                continue

            collected.append(block.copy())
            quiet_run = quiet_run + 1 if level < settings.silence_threshold else 0
            spoken_seconds = len(collected) * settings.block_seconds
            if quiet_run >= silence_blocks and spoken_seconds >= settings.min_seconds:
                break
            if len(collected) >= max_blocks:
                break

    if not collected:
        return b""
    audio = np.concatenate(collected, axis=0)
    return to_wav(audio.tobytes(), samplerate=settings.samplerate, channels=settings.channels)


def record_seconds(seconds: float, settings: MicSettings | None = None) -> bytes:
    """Record a fixed window — used by ``--mode enter`` and by ``doctor``."""
    settings = settings or MicSettings()
    sounddevice = _require_sounddevice()
    frames = int(seconds * settings.samplerate)
    try:
        recording = sounddevice.rec(
            frames,
            samplerate=settings.samplerate,
            channels=settings.channels,
            dtype="int16",
            device=settings.device,
        )
        sounddevice.wait()
    except Exception as exc:  # noqa: BLE001
        raise MicError(f"could not record from the input device: {exc}") from None
    return to_wav(recording.tobytes(), samplerate=settings.samplerate, channels=settings.channels)


def to_wav(pcm: bytes, *, samplerate: int, channels: int, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(samplerate)
        handle.writeframes(pcm)
    return buffer.getvalue()
