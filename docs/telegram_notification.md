# DAPL Telegram 通知功能 — 完整教程

本文件說明如何在 **DAPL Docker 容器**內設定 Telegram 單向通知，供訓練、優化等長時間任務完成時推播訊息到手機。

> **環境前提**
> - 容器名稱：`DAPL`
> - 專案根目錄（容器內）：`/workspace/DAPL`
> - 所有安裝、測試、執行皆應在容器內完成，或透過 `docker exec DAPL ...` 從 host 觸發
> - 容器內已具備 `Python 3` 與 `requests`，**無需額外 pip 安裝**

---

## 一、功能概述

### 設計目標

| 項目 | 說明 |
|------|------|
| 用途 | 任務開始／完成／失敗時，發送文字通知到個人 Telegram |
| 模式 | **單向發送（Outbound-only）** — 只呼叫 `sendMessage`，不接收使用者指令 |
| 憑證存放 | 容器內 `/workspace/DAPL/.env`（`chmod 600`，Git 忽略） |
| 整合方式 | 其他 Python 腳本 import 共用模組，或直接用 CLI 發送 |

### 架構流程

```
訓練 / 優化腳本
       │
       ▼
tools/telegram_notify.py   ← 讀取 .env、驗證、限速、脫敏
       │
       ▼
Telegram Bot API (HTTPS)
       │
       ▼
你的 Telegram 客戶端（手機 / 桌面）
```

### 相關檔案路徑

| 路徑 | 說明 |
|------|------|
| `tools/telegram_notify.py` | 核心通知模組與 CLI 入口 |
| `scripts/telegram_secure_setup.sh` | 一次性安全初始化（寫入 `.env`、刪除 webhook、發測試訊息） |
| `scripts/telegram_demo.py` | 整合示範（模擬任務開始 → 完成） |
| `docs/telegram_notification.md` | 本教程（含環境變數範本，見第七章） |
| `/workspace/DAPL/.env` | **實際憑證檔**（容器內，不進 Git） |

---

## 二、建立 Telegram Bot

### 步驟 1：向 BotFather 申請 Bot

1. 在 Telegram 搜尋 **@BotFather**
2. 傳送 `/newbot`
3. 依提示設定顯示名稱與 username（須以 `bot` 結尾）
4. 完成後會收到 **HTTP API Token**（格式類似 `數字:英數字串`）

請妥善保存 Token，並**勿貼到聊天、Issue、Email 或任何會被版本控制的檔案**。

### 步驟 2：取得 Chat ID

通知需指定「發給誰」。個人使用建議取自己的 numeric ID：

1. 在 Telegram 搜尋 **@userinfobot**
2. 傳送任意訊息，Bot 會回覆你的 **ID**（純數字）
3. 另可對你的 Bot 傳一則訊息後，以瀏覽器查詢 `getUpdates` 確認（詳見 [Telegram Bot API 文件](https://core.telegram.org/bots/api)）

記下此 ID，後續寫入 `TELEGRAM_CHAT_ID`。

### 步驟 3（建議）：與 Bot 建立對話

對你的 Bot（例如 `t.me/<你的_bot_username>`）傳送一則訊息，確保之後 `sendMessage` 能成功送達。

---

## 三、在 DAPL 容器內完成設定

### 方式 A：安全初始化腳本（建議）

在 **host** 終端機先匯出密鑰（不要寫進 shell 歷史檔的話，可改用 `-i` 進容器手動輸入）：

```bash
export TELEGRAM_BOT_TOKEN='你的_BOT_TOKEN'
export TELEGRAM_CHAT_ID='你的_CHAT_ID'

docker exec -e TELEGRAM_BOT_TOKEN -e TELEGRAM_CHAT_ID DAPL \
  bash /workspace/DAPL/scripts/telegram_secure_setup.sh
```

腳本會自動完成：

1. 寫入 `/workspace/DAPL/.env`，權限設為 `600`
2. 呼叫 `deleteWebhook`（確保無對外 HTTP 端點）
3. 發送一則測試通知

成功時終端會顯示 `[setup] test message sent`，手機應收到測試訊息。

### 方式 B：手動建立 `.env`

```bash
docker exec -it DAPL bash
```

在容器內編輯 `/workspace/DAPL/.env`，內容結構參考本文件 **第七章環境變數範本**，需包含：

| 變數 | 說明 |
|------|------|
| `TELEGRAM_BOT_TOKEN` | BotFather 提供的 Token |
| `TELEGRAM_CHAT_ID` | 你的 numeric Chat ID |
| `TELEGRAM_ALLOWED_CHAT_IDS` | 允許發送的 ID 白名單（建議與 `TELEGRAM_CHAT_ID` 相同） |
| `TELEGRAM_RATE_LIMIT_PER_MIN` | 每分鐘最多發送則數（預設建議 `20`） |

設定完成後執行：

```bash
chmod 600 /workspace/DAPL/.env
```

並手動刪除 webhook（可選但建議）：

```bash
# 在容器內，先 source .env 再 curl deleteWebhook
```

### 方式 C：單次指令不寫檔（臨時測試）

```bash
docker exec \
  -e TELEGRAM_BOT_TOKEN='...' \
  -e TELEGRAM_CHAT_ID='...' \
  DAPL python3 /workspace/DAPL/tools/telegram_notify.py -m "測試訊息"
```

適合快速驗證，不建議作為長期方案（重啟後需重新傳入）。

---

## 四、日常使用

### CLI 發送通知

```bash
docker exec DAPL python3 /workspace/DAPL/tools/telegram_notify.py -m "訊息內容"
```

常用選項：

| 選項 | 說明 |
|------|------|
| `-m` / `--message` | 訊息正文（必填） |
| `--parse-mode Markdown` | 使用 Markdown 格式 |
| `--silent` | 靜音通知（客戶端不響鈴） |
| `--env-file` | 指定其他 env 檔（預設讀取專案根目錄 `.env`） |

### 示範腳本

```bash
docker exec DAPL python3 /workspace/DAPL/scripts/telegram_demo.py
```

### 整合到其他 Python 腳本

在容器內執行的腳本中，import 路徑請指向：

- **模組**：`tools/telegram_notify.py`
- **建議函式**：`send_telegram_message()`（一行發送）
- **進階類別**：`TelegramNotifier`（需更多控制時）

整合時建議：

- 長任務開始／結束／例外時各發一則
- 非必要流程使用 `fail_silently=True`，避免未設定 Telegram 時中斷訓練
- 測試階段使用 `fail_silently=False`，以便發現設定問題

---

## 五、安全注意事項

### 5.1 威脅模型（本專案採用的假設）

本實作為 **單向通知 Bot**，不處理使用者傳入的指令，也不架設 webhook。主要風險來自：

1. **Token 外洩** — 任何人持有 Token 皆可冒充你的 Bot 發訊息
2. **憑證檔被其他使用者讀取** — 同一主機上的其他帳號可能讀取 `.env`
3. **誤將密鑰 commit 進 Git** — 歷史紀錄難以完全清除
4. **Token 出現在 log / 錯誤訊息** — 除錯輸出意外暴露密鑰

### 5.2 已內建的安全措施

實作位於 `tools/telegram_notify.py`，包含：

| 措施 | 說明 |
|------|------|
| 單向發送 | 僅 `sendMessage`，初始化腳本會 `deleteWebhook` |
| Chat ID 白名單 | `TELEGRAM_ALLOWED_CHAT_IDS` 限制可發送對象 |
| 憑證與程式分離 | Token 只存在 `.env` 或 process 環境變數 |
| 錯誤脫敏 | API / HTTP 錯誤會遮蔽 Token 字串 |
| 發送限速 | 預設每分鐘最多 20 則，防止腳本 bug 狂發 |
| `.env` 權限檢查 | 若檔案可被 group/other 讀取會發出警告 |
| Token 格式驗證 | 啟動時檢查格式，避免誤設空值或錯誤字串 |

### 5.3 Git 與版本控制

以下路徑已在 `.gitignore` 排除，**不得 commit**：

| 路徑 | 說明 |
|------|------|
| `.env` | 實際 Bot Token 與 Chat ID |
| `.env.*` | 其他環境檔變體 |
| `config/telegram.local.env` | 本機覆寫用憑證檔 |
| `secrets/` | 通用密鑰目錄 |

可安全 commit 的檔案（不含真實密鑰）：

- `tools/telegram_notify.py`
- `scripts/telegram_secure_setup.sh`
- `scripts/telegram_demo.py`
- `docs/telegram_notification.md`（本文件）

### 5.4 操作守則（請務必遵守）

1. **不要把 Token 寫進** `.py`、`.json`、`.md`、README、筆記、聊天紀錄
2. **不要把 Token 貼到** Cursor 對話、Slack、GitHub Issue、Email
3. **若 Token 曾外洩**，立即至 @BotFather 執行 `/revoke` 輪替，並更新容器內 `.env`
4. **不要為此 Bot 開啟 webhook** 或公開 HTTP 服務接收 Telegram 推送
5. **不要實作「收到訊息就執行 shell」** — 這是常見的 Bot 劫持手法
6. **`.env` 權限維持 `600`**：`chmod 600 /workspace/DAPL/.env`
7. **僅發給自己的 Chat ID**，不要把 Bot 加入公開群組後用同一 Token 發敏感資訊
8. commit 前執行下方「驗證指令」確認無密鑰被追蹤

### 5.5 Token 外洩時的處理流程

1. @BotFather → `/revoke` → 選擇你的 Bot → 取得新 Token
2. 更新容器內 `/workspace/DAPL/.env` 的 `TELEGRAM_BOT_TOKEN`
3. 執行 `scripts/telegram_secure_setup.sh` 或手動 `deleteWebhook`
4. 發送測試訊息確認恢復正常
5. 檢查 Git 歷史與 log 是否曾記錄舊 Token（若有，視為已洩漏並完成輪替即可）

---

## 六、驗證與除錯

### 6.1 確認 Git 未追蹤憑證

在 host 專案目錄執行：

```bash
cd /path/to/DAPL   # 或 host 上對應的 DAPL 目錄

git check-ignore -v .env
git ls-files .env
git log -p --all -S '你的_BOT_TOKEN前幾碼' --
```

預期結果：

- `git check-ignore` 顯示 `.env` 被 ignore
- `git ls-files .env` 無輸出
- `git log` 搜尋 Token 無結果

### 6.2 確認 `.env` 權限

```bash
docker exec DAPL stat -c '%a %n' /workspace/DAPL/.env
```

預期：`600 /workspace/DAPL/.env`

### 6.3 確認 webhook 已關閉

初始化腳本執行後應顯示 webhook 已刪除。若需手動確認，可在容器內對 Bot API 查詢 `getWebhookInfo`（勿將 Token 貼到公開處）。

### 6.4 常見錯誤

| 現象 | 可能原因 | 處理方式 |
|------|----------|----------|
| `Telegram is not configured` | `.env` 不存在或變數未設 | 依第三章重新設定 |
| `chat not found` | 尚未對 Bot 傳過訊息 | 先手動對 Bot 傳一則訊息 |
| `Unauthorized` | Token 錯誤或已 revoke | 向 BotFather 取得新 Token 並更新 `.env` |
| `TELEGRAM_CHAT_ID is not in TELEGRAM_ALLOWED_CHAT_IDS` | 白名單與 Chat ID 不一致 | 將兩者設為相同 numeric ID |
| `rate limit exceeded` | 短時間發送過多 | 調整腳本邏輯或提高 `TELEGRAM_RATE_LIMIT_PER_MIN` |

---

## 七、環境變數參考

| 變數 | 必填 | 說明 |
|------|------|------|
| `TELEGRAM_BOT_TOKEN` | 是 | BotFather 提供的 HTTP API Token |
| `TELEGRAM_CHAT_ID` | 是 | 接收通知的 numeric Chat ID |
| `TELEGRAM_ALLOWED_CHAT_IDS` | 建議 | 逗號分隔白名單；預設行為等同只允許 `TELEGRAM_CHAT_ID` |
| `TELEGRAM_RATE_LIMIT_PER_MIN` | 否 | 每分鐘發送上限，預設 `20` |

### `.env` 範本

請勿填入真實 Token。建議寫入容器內 `/workspace/DAPL/.env` 並執行 `chmod 600`。

**建議做法（在 host 上，透過安全初始化腳本）：**

```bash
export TELEGRAM_BOT_TOKEN='從 @BotFather 取得'
export TELEGRAM_CHAT_ID='從 @userinfobot 取得'
docker exec -e TELEGRAM_BOT_TOKEN -e TELEGRAM_CHAT_ID DAPL \
  bash /workspace/DAPL/scripts/telegram_secure_setup.sh
```

**或手動在容器內建立 `/workspace/DAPL/.env`：**

```env
# Local secrets — never commit
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_CHAT_ID=<your_numeric_id>
TELEGRAM_ALLOWED_CHAT_IDS=<your_numeric_id>
TELEGRAM_RATE_LIMIT_PER_MIN=20
```

---

## 八、快速檢查清單

設定完成後，請逐項確認：

- [ ] 已透過 @BotFather 建立 Bot 並取得 Token
- [ ] 已透過 @userinfobot（或等效方式）取得 Chat ID
- [ ] 已對 Bot 傳送過至少一則訊息
- [ ] `/workspace/DAPL/.env` 已建立且 `chmod 600`
- [ ] `.env` **未**出現在 `git status` 的追蹤清單中
- [ ] 已執行安全初始化或手動 `deleteWebhook`
- [ ] 測試訊息已成功送達手機
- [ ] Token 未出現在任何 commit、文件或公開聊天中
- [ ] 若 Token 曾外洩，已完成 `/revoke` 輪替

---

## 九、延伸整合建議

若要在長時間訓練管線中使用，建議在以下時機呼叫 `tools/telegram_notify.py` 提供的介面：

1. **任務開始** — 記錄 run id、參數摘要
2. **任務成功** — 記錄耗時、主要指標
3. **任務失敗** — 記錄錯誤類型與簡短訊息（避免在 Telegram 傳送完整 stack trace 或路徑敏感資訊）

潛在整合點（依專案需求選用）：

- `tools/optimization_runner.py` — 優化批次完成／失敗
- `pretrain_VAEwC.py` — 預訓練結束
- `tools/update_running_report.py` — 階段性進度摘要

---

## 十、參考連結

- [Telegram Bot API 官方文件](https://core.telegram.org/bots/api)
- [BotFather](https://t.me/BotFather)
- 專案內核心模組：`tools/telegram_notify.py`

---

*文件版本：對應 DAPL 容器內 `/workspace/DAPL` 路徑結構。若容器掛載路徑不同，請將文中 `/workspace/DAPL` 替換為實際掛載點。*
