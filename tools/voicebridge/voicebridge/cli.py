"""Command line entry point: serve, listen, ask, doctor, devices, cursor-status."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .bridge import Bridge
from .config import TARGETS, Config
from .mic import MicError, MicSettings, list_input_devices, record_seconds, record_until_silence
from .router import parse
from .speech import SpeechError, play_audio

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voicebridge",
        description="Voice prompts from Meta smart glasses into Claude, ChatGPT, Perplexity or Cursor.",
    )
    parser.add_argument("--config", help="path to a JSON config overlay")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the webhook/HTTP gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")

    ask = sub.add_parser("ask", help="send one line of text through the router")
    ask.add_argument("text", nargs="+")
    ask.add_argument("--target", choices=TARGETS, help="assistant to use when the text doesn't name one")
    ask.add_argument("--channel", default="cli:local", help="conversation key (default cli:local)")
    ask.add_argument("--speak", action="store_true", help="also play the answer out loud")
    ask.add_argument("--text-style", action="store_true", help="skip the spoken-brevity system prompt")

    listen = sub.add_parser("listen", help="record from the glasses (paired as a Bluetooth headset)")
    listen.add_argument("--mode", choices=("vad", "enter"), default="vad")
    listen.add_argument("--device", help="input device index or name substring")
    listen.add_argument("--target", choices=TARGETS)
    listen.add_argument("--channel", default="mic:local")
    listen.add_argument("--seconds", type=float, default=12.0, help="clip length for --mode enter")
    listen.add_argument("--threshold", type=float, help="RMS gate, default 0.012")
    listen.add_argument("--silence", type=float, help="seconds of quiet that end a clip")
    listen.add_argument("--no-speak", action="store_true", help="don't read answers back")
    listen.add_argument(
        "--require-name",
        action="store_true",
        help="only act on lines that name an assistant, so ambient talk is ignored",
    )
    listen.add_argument("--once", action="store_true", help="handle a single utterance and exit")

    sub.add_parser("devices", help="list input devices")
    sub.add_parser("doctor", help="check configuration, keys and audio")

    status = sub.add_parser("cursor-status", help="poll a Cursor cloud agent")
    status.add_argument("agent_id")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config.load(args.config)

    if args.command == "serve":
        return _serve(config, args)
    if args.command == "devices":
        return _devices()
    if args.command == "doctor":
        return _doctor(config)
    if args.command == "ask":
        return _ask(config, args)
    if args.command == "listen":
        return _listen(config, args)
    if args.command == "cursor-status":
        return _cursor_status(config, args)
    parser.error(f"unknown command {args.command}")
    return 2


def _serve(config: Config, args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host or config.host
    port = args.port or config.port
    if not config.ingest_token:
        print(
            f"{DIM}note: VOICEBRIDGE_INGEST_TOKEN is unset, so /v1/* stays disabled "
            f"(webhooks still work){RESET}",
            file=sys.stderr,
        )
    if args.reload:
        uvicorn.run("voicebridge.server:create_app", host=host, port=port, factory=True, reload=True)
        return 0
    from .server import create_app

    uvicorn.run(create_app(config), host=host, port=port)
    return 0


def _ask(config: Config, args: argparse.Namespace) -> int:
    bridge = Bridge(config)
    exchange = bridge.handle_text(
        " ".join(args.text),
        args.channel,
        spoken=not args.text_style,
        target_override=args.target,
    )
    _print_exchange(exchange)
    if args.speak:
        _speak(bridge, exchange)
    return 0 if exchange.reply.kind != "error" else 1


def _listen(config: Config, args: argparse.Namespace) -> int:
    bridge = Bridge(config)
    ready, reason = bridge.transcriber.available()
    if not ready:
        print(f"{RED}speech-to-text unavailable:{RESET} {reason}", file=sys.stderr)
        return 1

    settings = MicSettings(device=_device(args.device))
    if args.threshold is not None:
        settings.silence_threshold = args.threshold
    if args.silence is not None:
        settings.silence_seconds = args.silence

    print(
        f"{DIM}Listening on {settings.device if settings.device is not None else 'the default input device'}. "
        f"Say e.g. “Claude, what's the fastest way to cut this loss curve?”. Ctrl-C to stop.{RESET}"
    )
    try:
        while True:
            try:
                if args.mode == "enter":
                    input(f"{DIM}press Enter, then speak…{RESET}")
                    audio = record_seconds(args.seconds, settings)
                else:
                    audio = record_until_silence(settings)
            except MicError as exc:
                print(f"{RED}microphone error:{RESET} {exc}", file=sys.stderr)
                return 1
            if not audio:
                continue

            try:
                transcript = bridge.transcriber.transcribe(audio, filename="clip.wav")
            except SpeechError as exc:
                print(f"{RED}transcription failed:{RESET} {exc}", file=sys.stderr)
                continue

            print(f"{DIM}heard:{RESET} {transcript}")
            if args.require_name and not parse(transcript).explicit_target:
                print(f"{DIM}(no assistant named, ignoring){RESET}")
                if args.once:
                    return 0
                continue

            exchange = bridge.handle_text(transcript, args.channel, target_override=args.target)
            _print_exchange(exchange)
            if not args.no_speak:
                _speak(bridge, exchange)
            if args.once:
                return 0
    except KeyboardInterrupt:
        print()
        return 0


def _speak(bridge: Bridge, exchange) -> None:
    rendered = bridge.speech_audio(exchange.reply)
    if rendered is None:
        return
    audio, filename, _ = rendered
    try:
        play_audio(audio, filename[filename.rfind(".") :])
    except SpeechError as exc:
        print(f"{DIM}(couldn't play audio: {exc}){RESET}", file=sys.stderr)


def _print_exchange(exchange) -> None:
    colour = RED if exchange.reply.kind == "error" else GREEN
    print(f"{colour}[{exchange.reply.target or 'bridge'}]{RESET} {exchange.reply.text}")
    if exchange.reply.speech and exchange.reply.speech != exchange.reply.text:
        print(f"{DIM}spoken: {exchange.reply.speech}{RESET}")


def _devices() -> int:
    try:
        devices = list_input_devices()
    except MicError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    if not devices:
        print("no input devices found")
        return 1
    for device in devices:
        print(f"{device['index']:>3}  {device['name']}  ({device['channels']} ch)")
    print(f"\n{DIM}Pick the one whose name mentions your glasses, then: voicebridge listen --device <index>{RESET}")
    return 0


def _doctor(config: Config) -> int:
    bridge = Bridge(config)
    diagnostics = bridge.diagnostics()
    print(f"default target: {diagnostics['default_target']}")
    print("\nassistants")
    for name, info in diagnostics["backends"].items():  # type: ignore[union-attr]
        mark = f"{GREEN}ready{RESET}" if info["ready"] else f"{RED}no{RESET}   "
        print(f"  {mark}  {name:<11} {'' if info['ready'] else info['reason']}")

    for label, key in (("speech-to-text", "speech_to_text"), ("text-to-speech", "text_to_speech")):
        info = diagnostics[key]  # type: ignore[index]
        mark = f"{GREEN}ready{RESET}" if info["ready"] else f"{RED}no{RESET}   "
        print(f"\n{label}\n  {mark}  {info.get('provider')} {info.get('model') or info.get('voice')} "
              f"{'' if info['ready'] else info['reason']}")

    print("\ningress")
    for name, enabled in diagnostics["ingress"].items():  # type: ignore[union-attr]
        mark = f"{GREEN}on {RESET}" if enabled else f"{DIM}off{RESET}"
        print(f"  {mark}  {name}")

    print("\naudio input")
    try:
        devices = list_input_devices()
        for device in devices[:8]:
            print(f"  {device['index']:>3}  {device['name']}")
        if not devices:
            print(f"  {DIM}none found{RESET}")
    except MicError as exc:
        print(f"  {DIM}{exc}{RESET}")

    print(f"\nstate: {diagnostics['state_dir']}")
    default_ready = diagnostics["backends"][diagnostics["default_target"]]["ready"]  # type: ignore[index]
    return 0 if default_ready else 1


def _cursor_status(config: Config, args: argparse.Namespace) -> int:
    from .backends.cursor import CursorBackend

    backend = CursorBackend(config.cursor)
    try:
        payload = backend.status(args.agent_id)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
