"""
Telegram notifications via the Bot API — uses `requests` only (no extra deps,
works on the Jetson's Python 3.8).

Setup (see deploy/README_DB.md):
  1. In Telegram, message @BotFather -> /newbot -> get the bot TOKEN.
  2. Message your new bot once, then visit
     https://api.telegram.org/bot<TOKEN>/getUpdates to find your chat id.
  3. Put token + chat id in monitor_config.json.
"""

from __future__ import annotations

import requests


def send_telegram(token: str, chat_id: str, message: str) -> tuple[bool, str]:
    """Send a message. Returns (ok, info). Never raises."""
    if not token or not chat_id:
        return False, "telegram not configured"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=10,
        )
        if r.status_code == 200:
            return True, "sent"
        return False, f"HTTP {r.status_code}: {r.text[:160]}"
    except Exception as e:
        return False, f"network error: {e}"


def get_updates(token: str, offset=None, timeout: int = 0):
    """Poll incoming messages (for the bot's command listener). Short poll
    (timeout=0) so it doesn't block the worker loop. Returns (updates_list, ok);
    never raises. Pass offset = last_update_id + 1 to consume processed messages."""
    if not token:
        return [], False
    try:
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                         params=params, timeout=timeout + 10)
        if r.status_code != 200:
            return [], False
        d = r.json()
        return (d.get("result", []), True) if d.get("ok") else ([], False)
    except Exception:
        return [], False
