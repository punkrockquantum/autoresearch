# voicebridge

Talk to your Meta glasses, get an answer from **Claude**, **ChatGPT**, **Perplexity** or **Cursor**.

You say *"Claude, why would val bpb get worse when I raise the batch size?"* — the words reach one
gateway, which picks the assistant you named, keeps the conversation going, and sends the answer back
to whichever channel you spoke through, short enough to be read into your ear.

```
                       ┌──────────────── voicebridge gateway ────────────────┐
"Hey Meta, send a      │                                                    │
 message to Bridge" ──▶│ WhatsApp webhook ─┐                    ┌─▶ Claude   │
                       │                   │                    │  (API or  │
glasses as BT mic ────▶│ local mic + STT ──┼─▶ router ─▶ session ┤   CLI)    │
                       │                   │   (who?   (memory  ├─▶ ChatGPT  │
Siri / Shortcut ──────▶│ POST /v1/ingest ──┤    what?)  per      ├─▶ Perplexity
                       │                   │            channel)├─▶ Cursor   │
Telegram voice note ──▶│ Telegram webhook ─┘                    │  (cloud    │
                       │                                        │   agent)   │
                       │            answer ──▶ text + optional TTS voice note │
                       └──────────────────────────────────────────────────────┘
```

## Which route should you use?

Meta's glasses do not let a third party register a wake word, and the
[Wearables Device Access Toolkit](https://developers.meta.com/wearables/) (camera/mic access from
your own iOS/Android app) is still a developer preview with publishing targeted for 2026. So the
routes below are the ones that work today, in the order I'd try them.

| Route | Hands-free? | Setup | Best for |
|---|---|---|---|
| **A. WhatsApp** | Fully — glasses dictate and send, notification read-aloud brings the answer back | Meta Cloud API number + public HTTPS | Walking around, quick questions |
| **B. Glasses as Bluetooth mic** | Nearly — one keypress or voice-activated capture on a nearby machine | `pip install` and go | Long prompts, Cursor coding tasks, local Whisper, private audio |
| **C. iOS Shortcut / Android Tasker** | Almost — "Hey Siri, ask Claude", using the glasses as headset mic | One shortcut, one bearer token | Phone-only setups, no Business number |
| **D. Telegram** | Phone-side voice notes; answers come back as playable voice notes | Bot token, 2 minutes | Testing the pipeline, voice replies |

All four hit the same router, share one conversation memory per channel, and understand the same
spoken grammar. Turn on as many as you like.

### Route A — WhatsApp (the hands-free one)

The glasses can compose and send WhatsApp messages on voice command, so this needs no app of your
own. You message a *second* WhatsApp number that belongs to your bridge (a WhatsApp Cloud API test
number is free), the gateway answers, and the glasses read the incoming reply to you.

1. Create a Meta app → add **WhatsApp** → note the **test phone number id** and a temporary or
   system-user **access token**; copy the app's **App Secret**.
2. Expose the gateway over HTTPS (`cloudflared tunnel --url http://localhost:8808`, ngrok, or a small
   VPS). Meta requires a public HTTPS callback.
3. In **WhatsApp → Configuration**, set the callback URL to `https://<host>/webhook/whatsapp`, the
   verify token to your `WHATSAPP_VERIFY_TOKEN`, and subscribe to the **messages** field.
4. Add your own number to `VOICEBRIDGE_ALLOWED_WHATSAPP_SENDERS` and save that bridge number as a
   contact on your phone (e.g. "Bridge").
5. Say: **"Hey Meta, send a message to Bridge on WhatsApp"** → *"Claude, what's a good learning rate
   warmup for a 20 minute run?"*

Notes and limits, so nothing surprises you:

- Cloud API only lets a business reply inside a 24-hour window after your last message — fine here,
  since you always speak first.
- Test numbers allow a small allowlist of recipients; a verified number lifts that.
- The glasses dictate *text*, so transcription is Meta's. If you send an actual voice note instead,
  the gateway downloads and transcribes it itself.
- Hearing the answer relies on notification read-aloud being enabled for WhatsApp in the Meta AI app;
  otherwise ask the glasses to read your messages.

### Route B — glasses as a Bluetooth microphone (highest fidelity)

Pair the glasses to your laptop as a headset. They become a normal input device *and* the output
device, so the whole loop — your voice in, spoken answer out — happens in your glasses with no
messaging app in the middle.

```bash
voicebridge devices                      # find the glasses' input index
voicebridge listen --device 3            # voice-activated: talk, pause, get an answer
voicebridge listen --mode enter          # press Enter, then speak
voicebridge listen --require-name        # ignore anything that doesn't name an assistant
```

`listen` gates on loudness: speech starts a clip, ~1.1 s of quiet ends it (tune with `--threshold`
and `--silence`). With `VOICEBRIDGE_STT_PROVIDER=local` the audio never leaves the machine. This is
the route to use for Cursor: long, precise instructions about your repo.

Two caveats worth knowing: the glasses can be connected to the phone or the laptop, not usefully
both, and while Meta AI is itself recording, the mic is not yours.

### Route C — iOS Shortcut (or Android Tasker)

With the glasses acting as your phone's headset, "Hey Siri" hears you through them. A one-step
Shortcut posts the dictated text to the gateway and speaks the reply back. Recipe:
[`docs/ios-shortcut.md`](docs/ios-shortcut.md).

### Route D — Telegram

Easiest to stand up, and the only channel that reliably sends *voice* answers you can play through
the glasses. Create a bot with @BotFather, then:

```bash
curl "https://api.telegram.org/bot<TOKEN>/setWebhook" \
  -d url=https://<host>/webhook/telegram -d secret_token=<TELEGRAM_SECRET_TOKEN>
```

Send `/start`, read your chat id from the log line, put it in `VOICEBRIDGE_ALLOWED_TELEGRAM_CHATS`.

## Install

```bash
cd tools/voicebridge
uv venv && uv pip install -e ".[dev]"        # core + tests
uv pip install -e ".[mic]"                   # route B: microphone capture
uv pip install -e ".[local-stt]"             # optional: on-device Whisper
```

Set at least one assistant key and check the wiring:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...                 # ChatGPT, and the default STT/TTS
voicebridge doctor
voicebridge ask "claude what is bits per byte"
voicebridge serve
```

`voicebridge doctor` prints, per assistant and per channel, whether it is ready and what is missing
if not. Exit code is non-zero when your default assistant isn't usable.

## Speaking to it

Name the assistant first; everything after it is the prompt.

| You say | What happens |
|---|---|
| "Claude, why is my loss spiking?" | Claude answers, and stays the default |
| "Ask chat GPT to summarise the Muon optimizer" | ChatGPT answers |
| "Perplexity, what did the Fed do today?" | Web-grounded answer, sources in the text reply |
| "Cursor, add a unit test for the dataloader" | Cloud agent starts on your repo, link in the reply |
| "What's the capital of Peru?" | Goes to whoever you last named |
| "New chat" / "start over" | Clears the conversation, keeps the assistant |
| "Switch to Perplexity" | Changes the default |
| "…in detail" | Longer answer (default budget is ~90 spoken words) |
| "Briefly …" | Roughly one-third of that |
| "Repeat that" | Re-reads the last answer |
| "Status" / "Help" | Current assistant and turn count / the grammar above |

Transcribers mangle product names, so "cloud", "clawed", "chat gbt", "complexity" and "curser" all
resolve to the right assistant — matching is fuzzy on the name and strict about nothing else. Every
reply is shaped for the ear: markdown flattened, code blocks and URLs left out of the spoken version
but kept in the text one, and a word budget applied at a sentence boundary.

## Assistants

| Target | How it runs | Notes |
|---|---|---|
| `claude` | Messages API, or `VOICEBRIDGE_CLAUDE_MODE=cli` to run the Claude Code CLI in a checkout | CLI mode can read your actual repo — set `VOICEBRIDGE_CLAUDE_WORKSPACE` |
| `chatgpt` | `POST /v1/chat/completions` | Set `VOICEBRIDGE_OPENAI_MODEL` to a model your key can use |
| `perplexity` | `POST /v1/agent` with `tools: [{"type":"web_search"}]` | The Agent API, since Sonar `/chat/completions` is sunset on 2026-09-27; citations are appended to the text reply only |
| `cursor` | `POST /v0/agents` cloud agent (`api`), `cursor-agent` locally (`cli`), or append to a markdown queue (`queue`) | Spoken reply gives the status; the agent link goes in the text. Poll with `voicebridge cursor-status <id>` |

Cursor is an editor, not a chat endpoint, so "ask Cursor" means "start a coding agent". In `api` mode
set `VOICEBRIDGE_CURSOR_REPO=https://github.com/you/repo`; the agent opens a branch you review later.
`queue` mode needs no key at all — it just writes your spoken tasks to a file you open in Cursor.

## Configuration

Env-first; a JSON file at `$VOICEBRIDGE_CONFIG` (default `~/.config/voicebridge/config.json`) can
override any field. See [`.env.example`](.env.example) for the full list. The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `VOICEBRIDGE_DEFAULT_TARGET` | `claude` | Assistant used when you don't name one |
| `VOICEBRIDGE_INGEST_TOKEN` | — | Bearer token for `/v1/*`; **unset means those routes stay off** |
| `VOICEBRIDGE_SPOKEN_WORD_LIMIT` | `90` | Spoken answer budget |
| `VOICEBRIDGE_STT_PROVIDER` | `openai` | `openai`, `groq` (fastest), `local` (faster-whisper), `none` |
| `VOICEBRIDGE_TTS_PROVIDER` | `openai` | `openai`, `elevenlabs`, `piper`, `none` |
| `VOICEBRIDGE_HISTORY_TURNS` | `12` | Turns of context sent to the model |
| `VOICEBRIDGE_SESSION_TTL_MINUTES` | `180` | Idle gap after which context is dropped |
| `VOICEBRIDGE_STATE_DIR` | `~/.local/state/voicebridge` | Where conversations are persisted |

## HTTP API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | none | Liveness |
| `GET` | `/v1/status` | bearer | Readiness per assistant/channel, active conversations |
| `POST` | `/v1/ingest` | bearer | `{"text": …}` or `{"audio_base64": …}`; add `"speak": true` for audio back |
| `POST` | `/v1/ingest/audio` | bearer | Multipart upload of a clip |
| `GET`/`POST` | `/webhook/whatsapp` | Meta HMAC | Verification handshake / inbound messages |
| `POST` | `/webhook/telegram` | secret header | Inbound updates |

```bash
curl -s localhost:8808/v1/ingest -H "Authorization: Bearer $VOICEBRIDGE_INGEST_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text":"perplexity what happened in markets today","speaker":"simon"}' | jq .
```

`/v1/*` answers synchronously, so a Shortcut can speak the result. Webhooks acknowledge immediately
and answer in a background task — Meta and Telegram both retry a slow webhook, and a retried prompt
would otherwise be answered (and billed) twice.

## Security

This service turns inbound messages into paid API calls, so it is closed by default:

- **WhatsApp**: every payload is HMAC-verified against your App Secret, and the sender must be on
  `VOICEBRIDGE_ALLOWED_WHATSAPP_SENDERS`. An empty allowlist accepts nobody.
- **Telegram**: `X-Telegram-Bot-Api-Secret-Token` must match, and the chat must be allowlisted.
- **`/v1/*`**: constant-time bearer check; with no token configured the routes return 503 rather than
  running open.
- Uploads are capped at 25 MB; secrets never appear in `doctor`/`status` output.

Prompts and audio go to whichever vendor you route to. `VOICEBRIDGE_STT_PROVIDER=local` keeps
transcription on your machine, and `cursor` in `queue` mode keeps a task entirely local.

## Tests

```bash
uv run pytest            # ~60 tests, no network, no API keys needed
```

Vendor calls are intercepted, so the suite checks the request shapes (Anthropic's `system` +
`messages`, Perplexity's `input`/`instructions`/`tools`, Cursor's `prompt`/`source`), webhook
signature rejection, allowlisting, transcript echo, session persistence and the spoken-text shaping.

## Troubleshooting

- **`listen` never records** — wrong device (`voicebridge devices`), or the gate is too high for a
  Bluetooth mic: `--threshold 0.006`. If clips cut off mid-sentence, raise `--silence`.
- **Answers are too long to hear** — lower `VOICEBRIDGE_SPOKEN_WORD_LIMIT`; say "in detail" when you
  want the long version.
- **WhatsApp webhook verification fails** — the URL must be public HTTPS and `hub.verify_token` must
  match exactly; check `voicebridge serve` logs for the GET.
- **Everything goes to the wrong assistant** — say the name first and check `voicebridge ask "status"`;
  a name you speak always beats the stored default.
- **ChatGPT returns an empty answer** — a reasoning model spent the budget on thinking; raise
  `VOICEBRIDGE_OPENAI_MAX_TOKENS`.
