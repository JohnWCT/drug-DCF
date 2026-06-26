#!/usr/bin/env bash
# Secure one-time Telegram setup inside the DAPL container.
#
# Usage (from host — pass secrets via env, never commit them):
#   export TELEGRAM_BOT_TOKEN='...'
#   export TELEGRAM_CHAT_ID='137368494'
#   docker exec -e TELEGRAM_BOT_TOKEN -e TELEGRAM_CHAT_ID DAPL \
#     bash /workspace/DAPL/scripts/telegram_secure_setup.sh
#
# What this does:
#   1. Writes /workspace/DAPL/.env with chmod 600
#   2. Deletes any webhook (outbound-only bot, no inbound HTTP endpoint)
#   3. Sends a test notification

set -euo pipefail

ROOT="/workspace/DAPL"
ENV_FILE="${ROOT}/.env"

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
  echo "ERROR: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment." >&2
  exit 1
fi

umask 077
cat > "${ENV_FILE}" <<EOF
# Local secrets — never commit. Managed by scripts/telegram_secure_setup.sh
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
TELEGRAM_ALLOWED_CHAT_IDS=${TELEGRAM_CHAT_ID}
TELEGRAM_RATE_LIMIT_PER_MIN=20
EOF
chmod 600 "${ENV_FILE}"

echo "[setup] wrote ${ENV_FILE} (mode 600)"

# Remove webhook so nobody can point Telegram to a malicious URL with your token.
DELETE_URL="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/deleteWebhook?drop_pending_updates=true"
curl -fsS "${DELETE_URL}" >/dev/null
echo "[setup] webhook deleted (outbound-only mode)"

python3 "${ROOT}/tools/telegram_notify.py" -m "[DAPL] Telegram 通知已設定完成（安全模式）"
echo "[setup] test message sent to chat_id=${TELEGRAM_CHAT_ID}"
