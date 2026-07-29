from pathlib import Path
import re
t = Path(r"C:\Users\harsh\AppData\Local\nvm\v22.23.1\node_modules\apify-cli\dist\_register-CNQJunZH.js").read_text(encoding="utf-8", errors="replace")
# find gn= and _n=
for pat in [r"gn=\"[^\"]+\"", r"gn='[^']+'", r",gn=\"", r"gn=\"token\""]:
    print(pat, re.findall(pat, t)[:5])
# search near getToken for gn definition - look for const gn
for m in re.finditer(r".{0,30}gn=.?.token.?.{0,30}", t):
    if 'getToken' in t[max(0,m.start()-200):m.end()+50] or 'token' in m.group(0).lower():
        print(m.group(0)[:80])
# simpler: find all string constants near 'token' in credentials section
i = t.find("async function getToken")
print(t[i-500:i+300])
