# -*- coding: utf-8 -*-
"""Query the DeepSeek account balance so spend decisions are not guesswork."""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join("D:", os.sep, "Claude Pro",
                         "ai-agent-security-gateway", ".env"))

key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    raise SystemExit("DEEPSEEK_API_KEY not set")

r = requests.get("https://api.deepseek.com/user/balance",
                 headers={"Authorization": f"Bearer {key}"}, timeout=15)
print("status:", r.status_code)
try:
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception:
    print(r.text[:400])
