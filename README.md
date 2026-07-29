# Migration and Grief Narrative Analysis

Two Reddit corpora for semantic / sentiment / topic analysis:

| Corpus | Subreddits | Theme |
|--------|------------|--------|
| **A** | `r/expats`, `r/immigration`, `r/StudyAbroad` | Migration experiences |
| **B** | `r/GriefSupport`, `r/bereavement` | Loss / grief experiences |

**Rules:** post body only (no comments), &gt;100 words, last 3 years, soft flair preference, exclude visa/tax/paperwork Q&A from Corpus A.

## Setup

```powershell
cd D:\research_paper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Apify CLI is already used for auth. Either:

- stay logged in via `apify login`, or  
- set `$env:APIFY_TOKEN = "your_token"`

## Pipeline

### 1. Pilot scrape (cheap)

```powershell
python scripts/01_scrape_apify.py --mode pilot --corpus both --listings-only
```

### 2. Full scrape (costs Apify credits)

```powershell
# Preview planned runs
python scripts/01_scrape_apify.py --mode full --corpus both --dry-run

python scripts/01_scrape_apify.py --mode full --corpus a
python scripts/01_scrape_apify.py --mode full --corpus b
```

### 3. Filter & normalize

```powershell
python scripts/03_filter_and_normalize.py
```

### 4. NLP clean + lemmatize

```powershell
python scripts/04_preprocess_nlp.py
```

### Corpus status

Phase 1 output under `data/processed/` is frozen. A bounded-keyword audit found that
substring matching had incorrectly removed posts containing words such as `already`,
`first`, and `competition`. The corrected Phase 2-4 analysis corpus is written to
`data/analysis/`, not back into Phase 1:

- Corpus A: 906 migration posts, 123,066 lemma tokens
- Corpus B: 687 grief posts, 109,009 lemma tokens

Scraping remains paused. New collection must be treated as a new sampling frame and
must not be silently merged into the current prevalence corpus.

## Phase 2-4 analysis

Create and verify the dedicated environment:

```powershell
C:\Users\harsh\anaconda3\python.exe -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements.txt -r requirements-analysis.txt -c constraints-analysis.txt
.\.venv-analysis\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis-torch.txt -c constraints-analysis.txt
.\.venv-analysis\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv-analysis\Scripts\python.exe -m spacy download en_core_web_md
.\.venv-analysis\Scripts\python.exe scripts\check_env.py
```

Run the complete pipeline:

```powershell
.\.venv-analysis\Scripts\python.exe run_analysis.py
```

Reuse existing model outputs and regenerate only figures and the sanitized bundle. This range starts after annotation sampling, so existing manual coding is preserved:

```powershell
.\.venv-analysis\Scripts\python.exe run_analysis.py --from-step semantic-figures --replace-bundle
```

The pipeline uses four fixed seed-dictionary domains (Temporal, Identity, Absence,
and Digital), shared-basis PPMI profiles, document-level reliability and resampling,
seven-class emotion probabilities over sentence-clipped anchor windows, PPMI
co-occurrence networks, and a pooled NMF topic model selected by bootstrap stability.

Interpretation limits:

- Domain is the primary semantic unit; only 12 of 71 anchors pass the combined
  frequency and reliability gate.
- Digital full-dictionary results are dominated by polysemous anchors and require
  manual concordance coding. Its corrected emotion comparison is not significant.
- Emotion analysis is a window proxy for ABSA, not a trained aspect-specific model.
- Only 1 of 8 selected topics meets the recurrence threshold: the Study Abroad topic, larger in Corpus A.
- Co-occurrence networks are exploratory because many high-PPMI edges have low counts.

The only shareable archive is `outputs/research_analysis_bundle_reviewed.zip`.
Earlier bundle ZIPs are superseded and may contain row-level identifiers; do not distribute them.

Run tests and coverage:

```powershell
.\.venv-analysis\Scripts\python.exe -m pytest
```

## Outputs

| Path | Description |
|------|-------------|
| `data/raw/` | Raw Apify JSON dumps |
| `data/interim/corpus_{a,b}.parquet` | Filtered structured posts |
| `data/processed/corpus_{a,b}.parquet` | Final clean DataFrames |
| `data/processed/phase1_summary.json` | Counts & QC metrics |

Final DataFrame columns include: `post_id`, `corpus`, `subreddit`, `title`, `body`, `text`, `text_clean`, `tokens`, `created_at`, `url`, `flair`, `word_count`, `word_count_clean`.

## Config

- `config/filters.yaml` — date, word count, flairs, exclude keywords  
- `config/scrape_runs.yaml` — Apify actor, pilot/full limits, listing sorts, search seeds  

## Fire-and-forget (Apify-only; close your laptop)

Starts **only unfinished** search jobs on Apify (skips run names that already have
≥10 posts locally so you do **not** re-pay for the ~750 posts already scraped).

```powershell
# Preview what will be skipped vs started
python scripts/05_fire_and_forget.py --dry-run

# Start remaining jobs on Apify, then exit (safe to close laptop)
# Uses 1024MB/run + 90s stagger to stay under the ~16GB concurrent memory cap.
# If Console still shows many RUNNING jobs, wait or abort them first.
python scripts/05_fire_and_forget.py --min-items 10 --memory 1024 --stagger-seconds 90

# Laptop-only sequential (safer on free-tier memory; keep PC awake)
python scripts/01_scrape_apify.py --mode full --corpus both --engine trudax --searches-only --resume --workers 1 --timeout 1200

# Later — download cloud datasets into data/raw/
python scripts/02_download_datasets.py

# Then filter + NLP as usual
python scripts/03_filter_and_normalize.py
python scripts/04_preprocess_nlp.py
```

Manifest of `run_id` / `dataset_id`: `data/raw/apify_fire_forget_manifest.json`

## Cost note & pilot findings

**Primary actor:** `labrat011/reddit-scraper` (~$1.50 / 1k results, HTTP/HTML, includes flair).  
**Fallback:** `trudax/reddit-scraper-lite` (~$3.40 / 1k; Puppeteer; slower and more Reddit 403s).

| Finding | Detail |
|---------|--------|
| Free tier cap | `labrat011` free plan ≈ **25 results per run** — need paid Apify / higher plan for 10k×2 |
| Word-count yield | Pilot ≈ **70–75%** of posts had body &gt; 100 words (text posts) |
| Flair | Available on labrat output (`Partner Loss`, `Mom Loss`, etc.) — sparse but useful |
| No comments | `includeComments: false` / `skipComments: true` |

Always run **pilot** first. Full oversample toward 10k clean posts per corpus will require **many runs** (multi-sort + search) and a paid Apify plan.

### Optional: set API token for Python client

```powershell
$env:APIFY_TOKEN = "apify_api_..."   # from console.apify.com → Settings → Integrations
```

Without a token, the scraper falls back to `apify call` (uses your CLI login / keyring).

## Pilot validation (completed)

- Scraped sample from `r/GriefSupport` and `r/expats` via Apify CLI  
- Filter pipeline produced interim parquets  
- NLP pipeline produces `data/processed/corpus_{a,b}.parquet` with `text_clean` + `tokens`

Scale to 10k/corpus after confirming Apify plan limits and budget.
