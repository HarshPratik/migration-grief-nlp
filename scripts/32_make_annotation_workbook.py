"""Build an easy coding pack from the 150-window sample.

Writes:
  annotation/emotion_windows_150.xlsx   — Excel with dropdowns
  annotation/emotion_windows_150.html   — browser form (saves locally, export CSV)

Model guesses are kept off the coding sheet so they do not bias you.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "tables" / "p3_22_emotion_annotation_sample150.csv"
OUT_DIR = ROOT / "annotation"
XLSX = OUT_DIR / "emotion_windows_150.xlsx"
HTML = OUT_DIR / "emotion_windows_150.html"

CATEGORIES = [
    "aspect_owned",
    "surrounding_context",
    "ambiguous",
    "wrong_anchor_match",
]
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
CONF = ["1", "2", "3"]

INSTRUCTIONS = [
    ("What this is",
     "150 short snippets from the 10k Reddit posts. Each snippet is centred on one seed word (the “anchor”). You are checking whether our emotion model is looking at the right thing."),
    ("How long",
     "About 45–90 minutes. You only fill four columns: category, emotion, confidence, and optional notes. Skip nothing if you can; “ambiguous” is a valid category."),
    ("Where to type",
     "Use the sheet CODE HERE. Yellow cells are unfinished. Dropdowns are already set. Do not edit the grey columns."),
    ("1. Category (required)",
     "aspect_owned = the emotion is about the highlighted seed word itself (e.g. sadness about a funeral). surrounding_context = the snippet is emotional but not really about that word. ambiguous = you cannot tell. wrong_anchor_match = the word is used in a different sense (e.g. “loss” as money, “forgotten” as “I forgot to mention”)."),
    ("2. Emotion (required unless wrong_anchor_match)",
     "Pick the dominant feeling in the snippet: anger, disgust, fear, joy, neutral, sadness, surprise. If category is wrong_anchor_match, leave emotion blank or put neutral."),
    ("3. Confidence",
     "1 = guess, 2 = reasonably sure, 3 = clear. Use 1 freely; that is useful data."),
    ("4. Notes",
     "Optional. One short phrase is enough (e.g. “financial loss, not grief”)."),
    ("Do not",
     "Do not look at the MODEL GUESSES sheet until you have finished all 150 rows. Those labels are the computer’s answers for agreement later. Looking early will bias you."),
    ("Save",
     "Save this Excel file as you go. When done, send it back (the whole .xlsx). You can also use emotion_windows_150.html in a browser if you prefer clicking to Excel."),
]


def _load() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run scripts/22_annotation_sample.py --replace")
    df = pd.read_csv(CSV)
    df["window_text"] = df["window_text"].fillna("").astype(str)
    df["anchor"] = df["anchor"].fillna("").astype(str)
    return df


def _style_header(ws, row: int, cols: int, fill: str) -> None:
    font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
    bg = PatternFill("solid", fgColor=fill)
    for col in range(1, cols + 1):
        cell = ws.cell(row, col)
        cell.font = font
        cell.fill = bg
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def write_xlsx(df: pd.DataFrame) -> None:
    wb = Workbook()

    # --- instructions ---
    ws0 = wb.active
    ws0.title = "START HERE"
    ws0["A1"] = "Emotion coding — 150 windows"
    ws0["A1"].font = Font(bold=True, size=18, name="Calibri")
    ws0["A2"] = "Open the sheet “CODE HERE” and fill the yellow columns. Save often."
    ws0["A2"].font = Font(size=12, italic=True, color="333333")
    ws0.merge_cells("A1:B1")
    ws0.merge_cells("A2:B2")
    ws0["A4"] = "Topic"
    ws0["B4"] = "What to do"
    _style_header(ws0, 4, 2, "1F4E79")
    for i, (title, body) in enumerate(INSTRUCTIONS, start=5):
        ws0.cell(i, 1, title).font = Font(bold=True, name="Calibri")
        ws0.cell(i, 2, body).alignment = Alignment(wrap_text=True, vertical="top")
        ws0.row_dimensions[i].height = 48
    ws0.column_dimensions["A"].width = 28
    ws0.column_dimensions["B"].width = 110
    ws0.freeze_panes = "A5"
    ws0.row_dimensions[1].height = 24

    # --- coding sheet ---
    ws = wb.create_sheet("CODE HERE", 1)
    headers = [
        "id",
        "corpus",
        "domain",
        "seed word",
        "snippet (read this)",
        "category",
        "emotion",
        "confidence 1–3",
        "notes (optional)",
        "progress",
    ]
    for col, h in enumerate(headers, start=1):
        ws.cell(1, col, h)
    _style_header(ws, 1, len(headers), "1F4E79")
    ws.row_dimensions[1].height = 22

    yellow = PatternFill("solid", fgColor="FFF2CC")
    grey = PatternFill("solid", fgColor="F2F2F2")
    wrap = Alignment(wrap_text=True, vertical="center")
    thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    for i, row in df.iterrows():
        r = i + 2
        values = [
            row["annotation_id"],
            row["corpus"],
            row["domain"],
            row["anchor"].replace("_", " "),
            row["window_text"],
            "",
            "",
            "",
            "",
        ]
        for col, val in enumerate(values, start=1):
            cell = ws.cell(r, col, val)
            cell.border = thin
            cell.alignment = wrap
            if col in (6, 7, 8, 9):
                cell.fill = yellow
            else:
                cell.fill = grey
        # progress formula: 1 if category filled
        prog = ws.cell(r, 10, f'=IF(F{r}="","todo","done")')
        prog.border = thin
        prog.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[r].height = 36

    n = len(df)
    last = n + 1
    ws.cell(n + 3, 5, "Rows finished:")
    ws.cell(n + 3, 5).font = Font(bold=True)
    ws.cell(n + 3, 6, f'=COUNTIF(J2:J{last},"done")&" / {n}"')

    widths = {"A": 8, "B": 10, "C": 12, "D": 16, "E": 62, "F": 22, "G": 14, "H": 16, "I": 28, "J": 12}
    for letter, width in widths.items():
        ws.column_dimensions[letter].width = width

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:J{last}"

    dv_cat = DataValidation(type="list", formula1='"' + ",".join(CATEGORIES) + '"', allow_blank=True)
    dv_emo = DataValidation(type="list", formula1='"' + ",".join(EMOTIONS) + '"', allow_blank=True)
    dv_conf = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
    for dv, col in ((dv_cat, "F"), (dv_emo, "G"), (dv_conf, "H")):
        dv.error = "Pick from the list"
        dv.errorTitle = "Invalid"
        dv.prompt = "Use the dropdown"
        dv.showErrorMessage = True
        dv.showInputMessage = True
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")

    todo_fill = PatternFill("solid", fgColor="F4CCCC")
    done_fill = PatternFill("solid", fgColor="D9EAD3")
    ws.conditional_formatting.add(
        f"J2:J{last}",
        FormulaRule(formula=['J2="todo"'], fill=todo_fill),
    )
    ws.conditional_formatting.add(
        f"J2:J{last}",
        FormulaRule(formula=['J2="done"'], fill=done_fill),
    )

    table = Table(displayName="CodingTable", ref=f"A1:J{last}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(table)

    # --- model guesses (open later) ---
    ws2 = wb.create_sheet("MODEL GUESSES — open last")
    ws2["A1"] = "Do not use this sheet until the 150 rows are coded."
    ws2["A1"].font = Font(bold=True, color="9C0006", size=14)
    ws2.merge_cells("A1:F1")
    model_headers = ["id", "seed word", "snippet", "model emotion", "model confidence", "post_id"]
    for col, h in enumerate(model_headers, start=1):
        ws2.cell(3, col, h)
    _style_header(ws2, 3, 6, "833C0C")
    for i, row in df.iterrows():
        r = i + 4
        ws2.cell(r, 1, row["annotation_id"])
        ws2.cell(r, 2, str(row["anchor"]).replace("_", " "))
        ws2.cell(r, 3, row["window_text"]).alignment = Alignment(wrap_text=True)
        ws2.cell(r, 4, row.get("model_emotion", ""))
        ws2.cell(r, 5, row.get("model_p_max", ""))
        ws2.cell(r, 6, row.get("post_id", ""))
        ws2.row_dimensions[r].height = 32
    for letter, width in {"A": 8, "B": 16, "C": 70, "D": 16, "E": 18, "F": 14}.items():
        ws2.column_dimensions[letter].width = width
    ws2.freeze_panes = "A4"

    ws0.sheet_view.showGridLines = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX)


def write_html(df: pd.DataFrame) -> None:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": row["annotation_id"],
                "corpus": row["corpus"],
                "domain": row["domain"],
                "anchor": str(row["anchor"]).replace("_", " "),
                "text": row["window_text"],
                "post_id": str(row["post_id"]),
            }
        )
    payload = json.dumps(records, ensure_ascii=False)
    cats = json.dumps(CATEGORIES)
    emos = json.dumps(EMOTIONS)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Code 150 emotion windows</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#5c5c5c; --line:#e6e6e6; --bg:#f7f6f3; --card:#fff; --accent:#1f4e79; --warn:#9c4a00; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font: 16px/1.45 Georgia, "Times New Roman", serif; color:var(--ink); background:var(--bg); }}
  header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line); padding:12px 20px; z-index:2; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .meta {{ color:var(--muted); font-size:14px; }}
  .bar {{ height:8px; background:#eee; border-radius:99px; margin-top:8px; overflow:hidden; }}
  .bar > span {{ display:block; height:100%; background:var(--accent); width:0%; }}
  main {{ max-width: 880px; margin: 20px auto 80px; padding: 0 16px; }}
  .how {{ background:#fff; border:1px solid var(--line); padding:16px 18px; margin-bottom:16px; }}
  .how h2 {{ margin:0 0 8px; font-size:16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); padding:16px 18px; margin:12px 0; }}
  .card.done {{ border-color:#b7d3b0; }}
  .kicker {{ font-size:13px; color:var(--muted); letter-spacing:.02em; text-transform:uppercase; }}
  .seed {{ font-weight:bold; color:var(--accent); }}
  .snip {{ font-size:18px; margin:10px 0 14px; }}
  .row {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
  select, input, textarea, button {{ font: 14px/1.3 system-ui, sans-serif; }}
  select, input[type=text] {{ padding:6px 8px; min-width: 160px; }}
  textarea {{ width:100%; min-height:52px; margin-top:8px; }}
  button {{ background:var(--accent); color:#fff; border:0; padding:8px 14px; cursor:pointer; }}
  button.secondary {{ background:#fff; color:var(--accent); border:1px solid var(--accent); }}
  footer {{ position:sticky; bottom:0; background:#fff; border-top:1px solid var(--line); padding:10px 20px; display:flex; gap:10px; }}
  .warn {{ color:var(--warn); font-size:13px; }}
</style>
</head>
<body>
<header>
  <h1>Code 150 emotion windows</h1>
  <div class="meta">Fill category + emotion for each snippet. Progress is saved in this browser. Export when finished.</div>
  <div class="bar"><span id="bar"></span></div>
  <div class="meta" id="count">0 / 150 done</div>
</header>
<main>
  <details class="how" open>
    <summary><strong>How to code (1 minute)</strong></summary>
    <p><b>category</b> — <i>aspect_owned</i>: the feeling is about the seed word. <i>surrounding_context</i>: emotional, but not really about that word. <i>ambiguous</i>: cannot tell. <i>wrong_anchor_match</i>: the word is used in another sense (money “loss”, “forgotten to mention”).</p>
    <p><b>emotion</b> — anger, disgust, fear, joy, neutral, sadness, surprise. If the match is wrong, leave emotion blank or choose neutral.</p>
    <p><b>confidence</b> — 1 guess, 2 fairly sure, 3 clear.</p>
    <p class="warn">Do not open the Excel sheet “MODEL GUESSES” (or any computer labels) until you finish. That would bias you.</p>
  </details>
  <div id="cards"></div>
</main>
<footer>
  <button type="button" id="export">Download my coding (CSV)</button>
  <button type="button" class="secondary" id="clear">Reset this browser copy</button>
  <span class="meta">File stays on your computer. Nothing is uploaded.</span>
</footer>
<script>
const ITEMS = {payload};
const CATS = {cats};
const EMOS = {emos};
const KEY = "emotion_windows_150_v1";
const CONF = ["","1","2","3"];

function load() {{
  try {{ return JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ return {{}}; }}
}}
function save(state) {{ localStorage.setItem(KEY, JSON.stringify(state)); }}

const state = load();
const root = document.getElementById("cards");

function optionList(values, selected) {{
  return values.map(v => `<option value="${{v}}" ${{v===selected?"selected":""}}>${{v||"(pick)"}}</option>`).join("");
}}

function render() {{
  root.innerHTML = ITEMS.map((it, i) => {{
    const s = state[it.id] || {{}};
    const done = Boolean(s.category);
    return `<article class="card ${{done?"done":""}}" data-id="${{it.id}}">
      <div class="kicker">${{it.id}} · corpus ${{it.corpus}} · ${{it.domain}}</div>
      <div>Seed word: <span class="seed">${{it.anchor}}</span></div>
      <div class="snip">“${{it.text}}”</div>
      <div class="row">
        <label>category
          <select data-field="category">${{optionList([""].concat(CATS), s.category||"")}}</select>
        </label>
        <label>emotion
          <select data-field="emotion">${{optionList([""].concat(EMOS), s.emotion||"")}}</select>
        </label>
        <label>confidence
          <select data-field="confidence">${{optionList(CONF, s.confidence||"")}}</select>
        </label>
      </div>
      <textarea data-field="notes" placeholder="optional note">${{s.notes||""}}</textarea>
    </article>`;
  }}).join("");
  bind();
  progress();
}}

function bind() {{
  root.querySelectorAll("select, textarea").forEach(el => {{
    el.addEventListener("input", () => {{
      const card = el.closest(".card");
      const id = card.dataset.id;
      state[id] = state[id] || {{}};
      state[id][el.dataset.field] = el.value;
      save(state);
      card.classList.toggle("done", Boolean(state[id].category));
      progress();
    }});
  }});
}}

function progress() {{
  const n = ITEMS.filter(it => state[it.id] && state[it.id].category).length;
  document.getElementById("count").textContent = n + " / " + ITEMS.length + " done";
  document.getElementById("bar").style.width = (100 * n / ITEMS.length) + "%";
}}

document.getElementById("export").onclick = () => {{
  const header = ["annotation_id","corpus","domain","anchor","window_text","manual_category","manual_emotion","confidence_1_to_3","coder_notes","post_id"];
  const lines = [header.join(",")];
  for (const it of ITEMS) {{
    const s = state[it.id] || {{}};
    const cells = [it.id, it.corpus, it.domain, it.anchor, it.text, s.category||"", s.emotion||"", s.confidence||"", s.notes||"", it.post_id];
    lines.push(cells.map(c => `"${{String(c).replaceAll('"','""')}}"`).join(","));
  }}
  const blob = new Blob([lines.join("\\n")], {{type:"text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "emotion_windows_150_coded.csv";
  a.click();
}};

document.getElementById("clear").onclick = () => {{
  if (confirm("Erase coding saved in this browser?")) {{ localStorage.removeItem(KEY); location.reload(); }}
}};

render();
</script>
</body>
</html>
"""
    # Fix accidental double-escaping of JS template: the f-string already
    # doubled braces. The ITEMS JSON must not go through html.escape.
    HTML.write_text(html, encoding="utf-8")


def main() -> int:
    df = _load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_xlsx(df)
    write_html(df)
    print(f"Excel  {XLSX}")
    print(f"Browser {HTML}")
    print("Code the yellow columns (or the HTML form). Leave MODEL GUESSES closed until the end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
