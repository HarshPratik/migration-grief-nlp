from pathlib import Path
import re
t = Path(r"C:\Users\harsh\AppData\Local\nvm\v22.23.1\node_modules\apify-cli\dist\_register-CNQJunZH.js").read_text(encoding="utf-8", errors="replace")
for m in re.finditer(r"getKeyringEntry\([^)]{0,100}\)", t):
    print(m.group(0)[:120])
i = t.find("function getKeyringEntry")
print("---def---")
print(t[i:i+800])
i = t.find("KEYRING_SERVICE")
print("---service---", t[i:i+200])
i = t.find("apify-cli")
print("apify-cli idx", i, t[i-20:i+40] if i>=0 else "")
