# Running it somewhere permanent

The WhatsApp route needs a public HTTPS callback; the others don't. Pick the lightest thing that
covers the routes you actually use.

## Laptop plus a tunnel (quickest)

```bash
voicebridge serve                       # listens on :8808
cloudflared tunnel --url http://localhost:8808
```

Use the printed `https://…trycloudflare.com` URL as the WhatsApp callback (`/webhook/whatsapp`) and
as the Shortcut endpoint. Free tunnel URLs change on restart, so a named tunnel (or ngrok with a
reserved domain) is worth it once this becomes routine.

## systemd on a small VPS

`/etc/systemd/system/voicebridge.service`:

```ini
[Unit]
Description=voicebridge
After=network-online.target
Wants=network-online.target

[Service]
User=voicebridge
WorkingDirectory=/opt/voicebridge
EnvironmentFile=/etc/voicebridge.env
ExecStart=/opt/voicebridge/.venv/bin/voicebridge serve
Restart=on-failure
RestartSec=5
# The state directory holds conversation history.
StateDirectory=voicebridge
Environment=VOICEBRIDGE_STATE_DIR=/var/lib/voicebridge
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

```bash
install -m 600 -o root .env /etc/voicebridge.env     # secrets, root-only
systemctl enable --now voicebridge
journalctl -u voicebridge -f
```

Put Caddy or nginx in front for TLS:

```
bridge.example.com {
    reverse_proxy localhost:8808
}
```

## Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
ENV VOICEBRIDGE_STATE_DIR=/state
VOLUME /state
EXPOSE 8808
CMD ["voicebridge", "serve"]
```

```bash
docker build -t voicebridge tools/voicebridge
docker run -d --env-file tools/voicebridge/.env -p 8808:8808 -v voicebridge-state:/state voicebridge
```

## Before you point the internet at it

- `VOICEBRIDGE_ALLOWED_WHATSAPP_SENDERS` and `VOICEBRIDGE_ALLOWED_TELEGRAM_CHATS` must list your own
  numbers/chats. Empty means "accept nobody", which is the safe failure, but check it — an open
  bridge is someone else spending your API budget.
- `WHATSAPP_APP_SECRET` must be set, or every webhook POST is rejected as unsigned.
- Use a long random `VOICEBRIDGE_INGEST_TOKEN`, and leave it unset if you aren't using `/v1/*`.
- `curl https://<host>/healthz` should be the only route that answers without credentials.
- `voicebridge doctor` on the box, after loading the env file, tells you what's live.

## Keeping an eye on it

Every exchange logs one line — channel, the first 80 characters heard, the assistant chosen, and the
kind of reply — which is enough to spot a misrouted phrase or a backend that started failing.
`GET /v1/status` returns per-assistant readiness plus active conversations. If prompts are somehow
being answered twice, check that your tunnel isn't also forwarding an older gateway: webhooks
acknowledge before answering precisely so Meta and Telegram don't retry and double-bill you.
