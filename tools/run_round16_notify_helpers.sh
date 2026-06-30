#!/usr/bin/env bash
# Shared Telegram helpers for Round 16 stage scripts.

r16_notify() {
  python3 tools/round16_telegram_notify.py "$@" || true
}
