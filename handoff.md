# Handoff: Research paper NLP pipeline (Phases 1-4)

**Last updated:** 2026-07-27  
**Workspace:** `D:\research_paper`  
**Status:** Scraping paused. Current Phase 2-4 analysis and reviewed deliverables are complete; see `README.md` for rerun commands and interpretation limits.

Important: only `outputs/research_analysis_bundle_reviewed.zip` is approved for sharing. Earlier bundle ZIPs are superseded and must not be distributed.

---

## 1. Research goal (4 phases)

### Phase 1 — Data collection & preprocessing ✅ (done on current data)
- Scrape ~10k–50k posts from two domain groups (targets not yet reached).
- Clean: stop-words, lemmatization, normalize.
- **Output:** Corpus A + Corpus B as structured DataFrames.

### Phase 2 — Vector space & semantic similarity (complete)
- Shared-basis PPMI profiles at domain and qualified-anchor level.
- Split-half reliability gate, attenuation-corrected cosine, post bootstrap and stratified permutation.
- **Output:** comparison tables plus PCA/t-SNE descriptive views.

### Phase 3 — Emotion windows & co-occurrence (complete)
- Sentence-clipped ±5-token anchor windows classified with a seven-class emotion model.
- Strict domains exclude polysemous anchors; one post per author enters inferential resampling.
- Pairwise ±5-token co-occurrence networks require pair count >=3 and use PPMI weights.
- **Output:** emotion tables/figures and GraphML networks.

### Phase 4 — Pooled topic modeling (complete)
- Pooled NMF with baseline-adjusted bootstrap stability selection.
- Current reviewed model selects k=8; only Topic 2 (Study Abroad) meets the core recurrence threshold.
- **Output:** topic terms, prevalence, stability and corpus-loading figures.

---

## 2. Corpora definition

| Corpus | Theme | Subreddits |
|--------|--------|------------|
| **A** | Migration / expat experiences | `r/expats`, `r/immigration`, `r/StudyAbroad` |
| **B** | Loss / grief experiences | `r/GriefSupport`, `r/bereavement` |

### Scraping rules (user-specified)
- **Posts only** (title + body) — **no comments**.
- Keep posts **> 100 words**.
- Prefer flairs: Vent, Personal Story, Loss (soft — missing flair kept).
- **Exclude** visa/tax/paperwork Q&A (Corpus A blacklist).
- Timeframe: **last ~3 years** from mid-2023 (`date_from: 2023-07-01`).
- Target volume: **~10,000 posts per corpus** after filters (not met yet).

---

## 3. Environment

| Item | Value |
|------|--------|
| OS | Windows |
| Project root | `D:\research_paper` |
| Python venv | `D:\research_paper\.venv` |
| Apify CLI | Installed; auth via OS keyring (`apify login`) |
| Last Apify account used | `lengthy_arrowroot` (had free usage remaining; was **not** $0 when false budget alerts fired) |
| Earlier Apify account | `mighty_monk` (switched away) |

```powershell
cd D:\research_paper
.\.venv\Scripts\Activate.ps1
```

Optional: set `APIFY_TOKEN` for `apify-client` without CLI keyring.

---

## 4. Project layout

```
research_paper/
├── config/
│   ├── filters.yaml              # dates, min words, flairs, exclude keywords
│   ├── scrape_runs.yaml          # actors, limits, search queries
│   └── pilot_*.json              # one-off Apify inputs
├── data/
│   ├── raw/
│   │   ├── corpus_a/             # local dump JSON per run
│   │   ├── corpus_b/
│   │   ├── cloud/                # bulk-downloaded Apify datasets
│   │   ├── pilot/
│   │   ├── seen_post_ids.json    # global dedup registry
│   │   ├── scrape_log_full_*.json
│   │   └── apify_*               # run lists, manifests, reports
│   ├── interim/                  # filtered (pre-NLP)
│   │   ├── corpus_a.parquet
│   │   ├── corpus_b.parquet
│   │   └── filter_summary.json
│   └── processed/                # FINAL Phase 1 tables
│       ├── corpus_a.parquet + .csv
│       ├── corpus_b.parquet + .csv
│       ├── phase1_summary.json
│       └── PHASE1_COMPLETE.md
├── scripts/
│   ├── 01_scrape_apify.py        # main orchestrator (parallel, resume, dedup)
│   ├── 02_download_datasets.py   # download from fire-and-forget manifest
│   ├── 02b_download_all_runs.py  # download from apify_runs_status_report.json
│   ├── 03_filter_and_normalize.py
│   ├── 04_preprocess_nlp.py
│   ├── 05_fire_and_forget.py     # start unfinished Apify jobs, exit
│   ├── 06_resurrect_timeouts.py  # resurrect TIMED-OUT runs
│   ├── scrape_plan.py            # shared plan + completed-run detection
│   └── seen_ids.py               # thread-safe post ID registry
├── docs/
│   ├── SPEED.md
│   └── HOME_RESUME.md
├── requirements.txt
├── README.md
└── handoff.md                    # this file
```

---

## 5. Phase 1 final outputs (as of last pipeline run)

### Counts after full filter + NLP

| Corpus | Posts | Subreddits |
|--------|------:|------------|
| **A** | **323** | expats 167, studyAbroad 89, immigration 67 |
| **B** | **687** | GriefSupport 590, bereavement 97 |
| **Total clean** | **1,010** | |

### Filter funnel (approx.)
| Stage | Count |
|-------|------:|
| Raw items loaded | ~4,200 |
| Normalized posts | ~4,120 |
| After date ≥ 2023-07-01 | ~3,494 |
| After word count > 100 | ~3,054 |
| After Corpus A exclude keywords | ~1,575 |
| After dedupe | **1,010** |

**Important:** Corpus A is small largely because the **visa/tax/paperwork blacklist** is aggressive (`exclude_keywords_corpus_a` in `config/filters.yaml`). Many immigration posts were dropped by design.

### Raw uniqueness (before strict filters)
- Registry / unique post IDs on disk: **~2,800**
- Raw rows across dumps (with overlap): **~4,200**

### Processed DataFrame columns
`post_id`, `corpus`, `subreddit`, `title`, `body`, `text`, `text_clean`, `tokens`, `created_at`, `url`, `flair`, `score`, `word_count`, `word_count_clean`, `username`, `flair_preferred`, `flair_missing`, …

- **`text`**: original title + body (analysis base for sentiment/windows).
- **`text_clean` / `tokens`**: lemmatized, stop-words removed (good for embeddings/topics).

### QC
- Empty `text_clean` rate: **0%**
- Min word count on filtered set: **> 100**
- Date range: ~2023-07 → 2026-07

---

## 6. Scraping stack & actors

### Primary actor
**`trudax/reddit-scraper-lite`**
- Pay-per-result (~$3.40 / 1k stored results historically; PPE pricing).
- Puppeteer-based; slow; frequent Reddit **403**s.
- Posts-only: `skipComments: true`, `maxComments: 0`.
- Default memory often **2048 MB** per run.

### Secondary (pilot only)
**`labrat011/reddit-scraper`**
- Faster HTTP-style; free tier hard-capped at **~25 results/run**.
- Schema uses `selftext` / `subreddit` / `type` (handled in normalize).

### Apify account limits (critical)
| Limit | Value |
|-------|------:|
| Concurrent actor memory | **16,384 MB** |
| Memory per trudax run (typical) | **2,048 MB** |
| **Max concurrent runs at 2 GB** | **8** |
| At 1 GB/run | up to **16** |

**Mistake already made:** staying at `workers=1` after a memory-full incident was over-conservative. Correct approach when headroom exists: **`workers=4`** (8 GB) or up to **6–7** if nothing else is running.

### Cost / speed reality
- Browser scraper + residential proxies → often **~15–45 s wall time per raw post**.
- Overlapping search queries → high **duplicate rate** (~30–40% of raw rows not new uniques).
- Dedup after download **does not save Apify money** — only cleans the research set.
- Sequential pace: roughly **~600 new uniques / 7 h** was observed; 10k×2 is multi-day with this actor alone.

---

## 7. Deduplication rules (do not re-scrape wastefully)

1. **`data/raw/seen_post_ids.json`** — global post ID set; reports raw/new/dup per run.
2. **Resume skip:** run_name is “done” only if a local dump has **≥ 10 items** (1-item timeout dumps are **retried**).
3. **Filter step:** `drop_duplicates(subset=["post_id"])` keep longest text.
4. **Never** re-launch the full plan from scratch — only **unfinished** `run_name`s (see `scripts/scrape_plan.py` / `05_fire_and_forget.py --dry-run`).

Local data under `data/raw/` is **kept** when switching to cloud-only workflows; cloud datasets are separate until downloaded.

---

## 8. How to run / resume

### Activate
```powershell
cd D:\research_paper
.\.venv\Scripts\Activate.ps1
```

### Scrape (when unpausing)
```powershell
# Recommended: 4 workers, 1h timeout, resume solid dumps
python scripts/01_scrape_apify.py --mode full --corpus both `
  --engine trudax --searches-only --resume --workers 4 --timeout 3600
```

### Fire-and-forget (start jobs on Apify, close laptop)
```powershell
python scripts/05_fire_and_forget.py --dry-run
python scripts/05_fire_and_forget.py --min-items 10 --memory 1024 --stagger-seconds 90
# later:
python scripts/02_download_datasets.py
```

### Reprocess Phase 1 after more scrapes
```powershell
python scripts/03_filter_and_normalize.py
python scripts/04_preprocess_nlp.py
```

### Resurrect timed-out Apify runs
```powershell
python scripts/06_resurrect_timeouts.py --dry-run
python scripts/06_resurrect_timeouts.py --limit 2
# or: apify runs resurrect <runId>
# Longer timeout on resurrect needs APIFY_TOKEN + API:
# POST /v2/actor-runs/{runId}/resurrect?timeout=3600
```

### Bulk download existing cloud datasets
```powershell
# needs data/raw/apify_runs_status_report.json from a status script
python scripts/02b_download_all_runs.py
```

---

## 9. Config knobs

### `config/filters.yaml`
- `date_from`, `min_word_count: 100`, `target_posts_per_corpus: 10000`
- `exclude_keywords_corpus_a` — **main reason Corpus A is small**
- `include_flairs` — soft preference only
- `subreddits` lists for A/B

### `config/scrape_runs.yaml`
- `primary_actor_id: trudax/reddit-scraper-lite`
- Search query batches for corpus_a / corpus_b
- `full.maxItems` / `maxPostCount` often **80** per run (finishes more reliably than 1000)

---

## 10. Known issues & lessons

1. **False “budget exhausted”** — crawler logs / CLI noise were misclassified as payment errors. Detector tightened; still verify Apify Billing before trusting a stop.
2. **Memory limit 16 GB** — do not start 8+ simultaneous 2 GB runs if others already RUNNING; check Console first.
3. **CLI `actors start` JSON** — sometimes non-zero exit with `"ok": true`; parser fixed in `05_fire_and_forget.py`.
4. **Timeouts at 1200s** — common mid-crawl; use **3600s**; resurrect continues request queue.
5. **UTF-8 BOM** on some PowerShell `Out-File` JSON — filter loader uses `utf-8-sig`.
6. **Windows Apify CLI** — invoke via `node …/apify.js` (see `resolve_apify_cmd` in `01_scrape_apify.py`), not bare `apify` from Python subprocess without path fix.
7. **labrat free tier** — 25 results/run; not enough for bulk.
8. **VPN on laptop** does not affect Apify cloud actors; residential proxy is in actor input.
9. **Processed interim files** were pilot-sized until full reprocess; always re-run 03+04 after big scrapes.

---

## 11. What is intentionally NOT done

- [ ] Reach 10k posts per corpus (scraping paused by design)  
- [x] Phase 2 embeddings / similarity — **done** (PPMI + BERT; see `docs/METHODS_RESULTS.md`)  
- [x] Phase 3 emotion windows / co-occurrence — **done**  
- [x] Phase 4 topic models — **done** (NMF k=8)  
- [ ] Apify-hosted orchestrator actor (discussed, not built)  
- [ ] Soften Corpus A blacklist (optional if A volume needed)  
- [ ] Manual coding of the 150-window annotation sample  
- [x] Word2Vec/FastText trained and evaluated empirically (`scripts/25_predictive_eval.py`); neighbour lists weak vs BERT at current N — re-run at 10k

---

## 12. Suggested next steps for a new session

1. **Smoke-test pipeline on current data**
   - Load `data/processed/corpus_*.parquet`
   - Quick SBERT or Word2Vec demo on `text_clean` / `text`
2. **If pipeline OK → scrape more**
   - New/diverse keywords; reduce query overlap  
   - `workers=4` (or 6 if 0 other runs)  
   - Re-run 03 + 04  
3. **If Corpus A too small after filters**
   - Review `exclude_keywords_corpus_a` (e.g. drop overly broad terms)  
   - Re-run 03 + 04 only (no re-scrape needed for that experiment)

---

## 13. Commands cheat sheet for new agent

```powershell
cd D:\research_paper
.\.venv\Scripts\Activate.ps1

# Phase 1 reprocess
python scripts/03_filter_and_normalize.py
python scripts/04_preprocess_nlp.py

# Scrape more (paused by user until pipeline validated)
python scripts/01_scrape_apify.py --mode full --corpus both --engine trudax --searches-only --resume --workers 4 --timeout 3600

# Apify status
apify info
apify runs ls trudax/reddit-scraper-lite --limit 20 --desc --json

# Inventory uniques
python scripts/_data_inventory.py   # if present; or recount from seen_post_ids.json
```

---

## 14. User preferences / decisions already made

- Prefer Apify CLI actors for Reddit collection.  
- Posts only, no comments.  
- Pause scraping until full pipeline is tested; then scale data.  
- Keep local raw dumps; don’t delete when moving orchestration.  
- Avoid re-scraping batches that already returned solid (≥10 item) dumps.  
- Account can switch when free usage ends — use `--resume` after new `apify login`.

---

*End of handoff. Point the next session at this file + `data/processed/PHASE1_COMPLETE.md` and `README.md`.*
