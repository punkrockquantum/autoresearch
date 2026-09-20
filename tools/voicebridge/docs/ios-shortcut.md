# Route C — "Hey Siri, ask Claude" through the glasses

With the glasses connected to your iPhone as a headset, Siri hears you through them and speaks the
answer back through them. A Shortcut is the glue: it dictates, posts to voicebridge, and reads the
reply. No Business number, no webhook, no public HTTPS if you stay on your own network or a VPN.

## Build the Shortcut

New Shortcut, named **Ask Claude** (the name is what you say to Siri):

1. **Dictate Text** — Language: your own; Stop Listening: *After Pause*.
2. **Get Contents of URL**
   - URL: `https://<your-host>/v1/ingest`
   - Method: `POST`
   - Headers:
     - `Authorization`: `Bearer <VOICEBRIDGE_INGEST_TOKEN>`
     - `Content-Type`: `application/json`
   - Request Body: **JSON**
     - `text` (Text) → the *Dictated Text* variable
     - `channel` (Text) → `ios`
     - `speaker` (Text) → your name, e.g. `simon`
3. **Get Dictionary Value** — Get `Value` for key `speech` in *Contents of URL*.
4. **Speak Text** — the *Dictionary Value*. Optionally set "Wait Until Finished".

Then: **"Hey Siri, Ask Claude"** → *"why would val bpb get worse when I raise the batch size?"*

The `speech` field is the ear-sized version of the answer; `reply` holds the full text (code blocks,
URLs, citations) if you'd rather show a notification than speak it.

## Variations

- **One shortcut per assistant.** Duplicate it and add `target` to the JSON body (`claude`,
  `chatgpt`, `perplexity`, `cursor`). Name them "Ask Perplexity", "Tell Cursor" and so on. A target
  you *speak* still wins over the `target` field, so "perplexity, …" inside "Ask Claude" works.
- **Let the bridge transcribe instead of Siri.** Use **Record Audio** → **Base64 Encode** and send
  `audio_base64` with `filename` set to `clip.m4a`. Useful when Siri's dictation mangles technical
  words: Whisper generally handles them better.
- **Voice that isn't Siri's.** Add `"speak": true` to the body, then **Get Dictionary Value** for
  `audio_base64` → **Base64 Decode** → **Play Sound**. That plays the TTS voice you configured.
- **Back Tap / Action Button.** Settings → Accessibility → Touch → Back Tap, or the Action Button on
  a 15 Pro and later, can launch the Shortcut without saying "Hey Siri" at all.
- **Android.** Tasker's *HTTP Request* action with the same JSON, triggered by a Bluetooth-connected
  profile or an assistant shortcut, then *Say* the `speech` field.

## Reaching your gateway from the phone

- Same Wi-Fi: `http://<laptop-ip>:8808` works, but the token then crosses your LAN in clear text.
- Anywhere: a Tailscale/WireGuard address, or a public HTTPS tunnel (see
  [`deploy.md`](deploy.md)). Prefer HTTPS whenever the bearer token leaves your own network.
