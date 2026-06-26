"""Send notifications via Telegram Bot API.

Security model (outbound-only):
- Bot token and chat id live in ``.env`` (gitignored) or process env vars.
- Messages are sent only to ``TELEGRAM_CHAT_ID`` (allowlist enforced).
- API errors and logs are redacted so tokens never appear in output.
- Optional rate limiting reduces abuse if another script spams ``send_message``.

Example (inside DAPL container)::

    python3 tools/telegram_notify.py --message "訓練完成"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LEN = 4096
_TOKEN_PATTERN = re.compile(r"\d{8,}:[A-Za-z0-9_-]{20,}")
_DEFAULT_RATE_LIMIT_PER_MIN = 20


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _redact_secrets(text: str, token: str = "") -> str:
    redacted = _TOKEN_PATTERN.sub("[REDACTED_TOKEN]", text)
    if token:
        redacted = redacted.replace(token, "[REDACTED_TOKEN]")
    return redacted


def _check_env_file_permissions(env_path: str) -> None:
    """Warn when secret file is readable by other users (POSIX only)."""
    if os.name != "posix" or not os.path.isfile(env_path):
        return
    mode = os.stat(env_path).st_mode & 0o777
    if mode & 0o077:
        print(
            f"[telegram_notify] warning: {env_path} mode={oct(mode)} is world/group "
            "readable; run: chmod 600 <file>",
            file=sys.stderr,
        )


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a dotenv-style file into ``os.environ``.

    Existing environment variables are not overwritten.
    """
    env_path = _resolve(path)
    if not os.path.isfile(env_path):
        return

    _check_env_file_permissions(env_path)

    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _parse_allowed_chat_ids(raw: str, fallback_chat_id: str) -> Set[str]:
    if raw.strip():
        return {part.strip() for part in raw.split(",") if part.strip()}
    return {fallback_chat_id}


def _validate_token(token: str) -> None:
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError("TELEGRAM_BOT_TOKEN format is invalid")


def _validate_chat_id(chat_id: str) -> None:
    if not re.fullmatch(r"-?\d+", chat_id):
        raise ValueError("TELEGRAM_CHAT_ID must be a numeric id")


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    allowed_chat_ids: Set[str]

    @classmethod
    def from_env(cls, *, env_file: str = ".env") -> Optional["TelegramConfig"]:
        load_env_file(env_file)
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return None

        _validate_token(token)
        _validate_chat_id(chat_id)

        allowed_raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
        allowed = _parse_allowed_chat_ids(allowed_raw, chat_id)
        if chat_id not in allowed:
            raise ValueError("TELEGRAM_CHAT_ID is not in TELEGRAM_ALLOWED_CHAT_IDS")

        return cls(bot_token=token, chat_id=chat_id, allowed_chat_ids=allowed)


def chunk_message(text: str, limit: int = MAX_MESSAGE_LEN) -> List[str]:
    """Split long text into Telegram-safe chunks."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = end
            end = split_at
        chunks.append(text[start:end])
        start = end
        if start < len(text) and text[start] == "\n":
            start += 1
    return chunks


class _SendRateLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max(1, max_per_minute)
        self._timestamps: List[float] = []

    def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        self._timestamps = [ts for ts in self._timestamps if ts >= cutoff]
        if len(self._timestamps) >= self.max_per_minute:
            raise RuntimeError(
                f"Telegram send rate limit exceeded ({self.max_per_minute}/min)"
            )
        self._timestamps.append(now)


class TelegramNotifier:
    """Thin wrapper around Telegram ``sendMessage`` API (outbound only)."""

    def __init__(
        self,
        config: TelegramConfig,
        *,
        timeout_sec: float = 15.0,
        rate_limit_per_minute: int = _DEFAULT_RATE_LIMIT_PER_MIN,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.timeout_sec = timeout_sec
        self._rate_limiter = _SendRateLimiter(rate_limit_per_minute)
        self.session = session or requests.Session()

    @classmethod
    def from_env(
        cls,
        *,
        env_file: str = ".env",
        timeout_sec: float = 15.0,
    ) -> Optional["TelegramNotifier"]:
        config = TelegramConfig.from_env(env_file=env_file)
        if config is None:
            return None
        limit_raw = os.environ.get(
            "TELEGRAM_RATE_LIMIT_PER_MIN", str(_DEFAULT_RATE_LIMIT_PER_MIN)
        )
        try:
            rate_limit = int(limit_raw)
        except ValueError:
            rate_limit = _DEFAULT_RATE_LIMIT_PER_MIN
        return cls(config, timeout_sec=timeout_sec, rate_limit_per_minute=rate_limit)

    @property
    def is_configured(self) -> bool:
        return bool(self.config.bot_token and self.config.chat_id)

    def _assert_allowed_chat(self, chat_id: str) -> None:
        if chat_id not in self.config.allowed_chat_ids:
            raise PermissionError("target chat_id is not in TELEGRAM_ALLOWED_CHAT_IDS")

    def _post_message(
        self,
        text: str,
        *,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> dict:
        chat_id = self.config.chat_id
        self._assert_allowed_chat(chat_id)
        self._rate_limiter.acquire()

        url = TELEGRAM_API_BASE.format(token=self.config.bot_token)
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_notification": disable_notification,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_sec)
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            msg = _redact_secrets(str(exc), self.config.bot_token)
            raise RuntimeError(f"Telegram HTTP error: {msg}") from exc

        if not body.get("ok"):
            safe_body = _redact_secrets(str(body), self.config.bot_token)
            raise RuntimeError(f"Telegram API error: {safe_body}")
        return body

    def send_message(
        self,
        text: str,
        *,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> List[dict]:
        """Send one or more chunked messages. Returns API responses."""
        if not text.strip():
            raise ValueError("message text is empty")

        responses: List[dict] = []
        for part in chunk_message(text):
            responses.append(
                self._post_message(
                    part,
                    parse_mode=parse_mode,
                    disable_notification=disable_notification,
                )
            )
        return responses

    def send_lines(
        self,
        lines: Iterable[str],
        *,
        header: str = "",
        parse_mode: Optional[str] = None,
    ) -> List[dict]:
        body = "\n".join(lines)
        text = f"{header}\n{body}".strip() if header else body
        return self.send_message(text, parse_mode=parse_mode)


def send_telegram_message(
    text: str,
    *,
    env_file: str = ".env",
    parse_mode: Optional[str] = None,
    disable_notification: bool = False,
    fail_silently: bool = False,
) -> bool:
    """Convenience helper for one-off notifications from other scripts."""
    try:
        notifier = TelegramNotifier.from_env(env_file=env_file)
    except Exception:
        if fail_silently:
            return False
        raise

    if notifier is None:
        if fail_silently:
            return False
        raise RuntimeError(
            "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID in /workspace/DAPL/.env (chmod 600)."
        )

    try:
        notifier.send_message(
            text,
            parse_mode=parse_mode,
            disable_notification=disable_notification,
        )
        return True
    except Exception as exc:
        if fail_silently:
            return False
        safe = _redact_secrets(str(exc))
        raise RuntimeError(safe) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a Telegram notification.")
    parser.add_argument("--message", "-m", required=True, help="Message body to send.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional dotenv file under project root (default: .env).",
    )
    parser.add_argument(
        "--parse-mode",
        choices=("Markdown", "MarkdownV2", "HTML"),
        default=None,
        help="Optional Telegram parse mode.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Send without notification sound on the client.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        send_telegram_message(
            args.message,
            env_file=args.env_file,
            parse_mode=args.parse_mode,
            disable_notification=args.silent,
            fail_silently=False,
        )
    except Exception as exc:
        print(f"[telegram_notify] failed: {exc}", file=sys.stderr)
        return 1

    print("[telegram_notify] message sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
