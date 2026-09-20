"""Speech in and speech out.

Transcription providers: an OpenAI-compatible endpoint (OpenAI itself or Groq,
which is noticeably faster), or faster-whisper running locally so audio never
leaves the machine. Synthesis: OpenAI, ElevenLabs, or a local piper binary.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import SpeechToTextConfig, TextToSpeechConfig
from .http import HttpError, describe_http_error, post_multipart, request_bytes

log = logging.getLogger(__name__)

_FORMAT_MIME = {
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
_FORMAT_EXT = {"opus": ".ogg", "ogg": ".ogg", "mp3": ".mp3", "aac": ".aac", "wav": ".wav", "pcm": ".pcm"}


class SpeechError(RuntimeError):
    pass


class Transcriber:
    def __init__(self, config: SpeechToTextConfig) -> None:
        self.config = config
        self._local_model = None

    def available(self) -> tuple[bool, str]:
        provider = self.config.provider
        if provider == "none":
            return False, "speech-to-text is disabled"
        if provider == "local":
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                return False, "faster-whisper isn't installed (pip install 'voicebridge[local-stt]')"
            return True, ""
        if not self.config.api_key:
            return False, f"no API key for the {provider} transcriber"
        return True, ""

    def transcribe(self, audio: bytes, *, filename: str = "audio.ogg") -> str:
        ready, reason = self.available()
        if not ready:
            raise SpeechError(reason)
        if self.config.provider == "local":
            return self._transcribe_local(audio, filename)
        return self._transcribe_remote(audio, filename)

    def _transcribe_remote(self, audio: bytes, filename: str) -> str:
        fields = {"model": self.config.model, "response_format": "json"}
        if self.config.language:
            fields["language"] = self.config.language
        try:
            payload = post_multipart(
                f"{self.config.base_url}/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                fields=fields,
                files={"file": (filename, audio, _guess_mime(filename))},
            )
        except HttpError as exc:
            raise SpeechError(describe_http_error(exc)) from None
        text = (payload.get("text") or "").strip()
        if not text:
            raise SpeechError("the transcriber returned no text")
        return text

    def _transcribe_local(self, audio: bytes, filename: str) -> str:
        from faster_whisper import WhisperModel

        if self._local_model is None:
            log.info("loading faster-whisper model %s", self.config.local_model)
            self._local_model = WhisperModel(
                self.config.local_model, compute_type=self.config.local_compute_type
            )
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio)
            temp_path = Path(handle.name)
        try:
            segments, _ = self._local_model.transcribe(
                str(temp_path), language=self.config.language or None, vad_filter=True
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            temp_path.unlink(missing_ok=True)
        if not text:
            raise SpeechError("nothing recognisable in that audio")
        return text


class Synthesizer:
    def __init__(self, config: TextToSpeechConfig) -> None:
        self.config = config

    def available(self) -> tuple[bool, str]:
        provider = self.config.provider
        if provider == "none":
            return False, "text-to-speech is disabled"
        if provider == "piper":
            if shutil.which(self.config.piper_path) is None:
                return False, "the piper binary isn't on PATH"
            if not self.config.piper_model:
                return False, "VOICEBRIDGE_PIPER_MODEL isn't set"
            return True, ""
        if not self.config.api_key:
            return False, f"no API key for the {provider} voice"
        return True, ""

    @property
    def mime(self) -> str:
        if self.config.provider == "piper":
            return "audio/wav"
        return _FORMAT_MIME.get(self.config.audio_format, "audio/ogg")

    @property
    def extension(self) -> str:
        if self.config.provider == "piper":
            return ".wav"
        return _FORMAT_EXT.get(self.config.audio_format, ".ogg")

    def synthesize(self, text: str) -> bytes:
        ready, reason = self.available()
        if not ready:
            raise SpeechError(reason)
        if self.config.provider == "piper":
            return self._piper(text)
        if self.config.provider == "elevenlabs":
            return self._elevenlabs(text)
        return self._openai(text)

    def _openai(self, text: str) -> bytes:
        body = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": text,
            "response_format": self.config.audio_format,
        }
        try:
            audio, _ = request_bytes(
                "POST",
                f"{self.config.base_url}/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                body=json.dumps(body).encode("utf-8"),
            )
        except HttpError as exc:
            raise SpeechError(describe_http_error(exc)) from None
        return audio

    def _elevenlabs(self, text: str) -> bytes:
        output_format = "opus_48000_64" if self.config.audio_format in {"opus", "ogg"} else "mp3_44100_128"
        url = (
            f"{self.config.base_url}/v1/text-to-speech/{self.config.voice}"
            f"?output_format={output_format}"
        )
        try:
            audio, _ = request_bytes(
                "POST",
                url,
                headers={"xi-api-key": self.config.api_key, "Content-Type": "application/json"},
                body=json.dumps({"text": text, "model_id": self.config.model}).encode("utf-8"),
            )
        except HttpError as exc:
            raise SpeechError(describe_http_error(exc)) from None
        return audio

    def _piper(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out_path = Path(handle.name)
        try:
            completed = subprocess.run(
                [self.config.piper_path, "--model", self.config.piper_model, "--output_file", str(out_path)],
                input=text,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                raise SpeechError((completed.stderr or "piper failed").strip().splitlines()[-1])
            return out_path.read_bytes()
        finally:
            out_path.unlink(missing_ok=True)


_PLAYERS = (
    ("afplay", []),           # macOS
    ("paplay", []),           # PulseAudio / PipeWire
    ("aplay", ["-q"]),        # ALSA (wav only)
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("mpv", ["--no-video", "--really-quiet"]),
)


def play_audio(audio: bytes, extension: str = ".ogg") -> None:
    """Play audio out of the default output device — i.e. the glasses."""
    player = next(((name, args) for name, args in _PLAYERS if shutil.which(name)), None)
    if player is None:
        raise SpeechError("no audio player found (install ffmpeg for ffplay, or mpv)")
    name, args = player
    if name == "aplay" and extension != ".wav":
        raise SpeechError("aplay only handles wav; set VOICEBRIDGE_TTS_FORMAT=wav or install ffmpeg")
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as handle:
        handle.write(audio)
        path = Path(handle.name)
    try:
        subprocess.run([name, *args, str(path)], check=False, capture_output=True)
    finally:
        path.unlink(missing_ok=True)


def _guess_mime(filename: str) -> str:
    return {
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".opus": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".amr": "audio/amr",
    }.get(Path(filename).suffix.lower(), "application/octet-stream")
