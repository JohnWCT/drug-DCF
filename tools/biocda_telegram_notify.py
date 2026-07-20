"""Telegram notifications for BioCDA architecture and training stages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.telegram_notify import send_telegram_message


def biocda_notify(message: str, *, fail_silently: bool = True) -> bool:
    prefix = "[BioCDA] "
    return send_telegram_message(prefix + message, fail_silently=fail_silently)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BioCDA Telegram notification")
    parser.add_argument("--message", "-m", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if Telegram not configured")
    args = parser.parse_args(argv)
    ok = biocda_notify(args.message, fail_silently=not args.strict)
    if not ok and args.strict:
        print("[biocda_telegram_notify] Telegram not configured", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
