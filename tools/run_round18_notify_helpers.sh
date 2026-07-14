#!/usr/bin/env bash
# Shared Telegram helpers for Round 18 stage scripts.

r18_notify() {
  python3 tools/round18_telegram_notify.py "$@" || true
}
