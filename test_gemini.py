import os
import google.generativeai as g
from dotenv import load_dotenv

load_dotenv()
k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print("KEY?", bool(k))
if not k:
    raise SystemExit("❌ No GEMINI_API_KEY found")

g.configure(api_key=k)
try:
    m = g.GenerativeModel("gemini-2.5-flash-lite")
    r = m.generate_content('Return ONLY this JSON: [{"ok": true}]')
    print("RESP?", bool(getattr(r, "text", "")))
    print("RAW OUTPUT:", (getattr(r, "text", "") or "")[:200])
except Exception as e:
    print("❌ Gemini call failed:", e)
