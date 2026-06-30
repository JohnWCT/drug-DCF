#!/usr/bin/env bash
# Shared Telegram helpers for Round 17 stage scripts.

r17_notify() {
  python3 tools/round17_telegram_notify.py "$@" || true
}
