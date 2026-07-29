# Migration and Grief Narrative Analysis

Comparative NLP study of two Reddit post corpora:

| Corpus | Theme | Subreddits |
|--------|--------|------------|
| **A** | Migration / expat experiences | `r/expats`, `r/immigration`, `r/StudyAbroad` |
| **B** | Loss / grief experiences | `r/GriefSupport`, `r/bereavement` |

Posts only (no comments), longer than 100 words, roughly the last three years from mid-2023, with soft flair preference. Corpus A drops visa/tax/paperwork Q&A via keyword filters.

The project supports data collection (Apify), cleaning, and Phases 2–4 analysis: semantic comparison (PPMI, BERT, Word2Vec, FastText), emotion windows, co-occurrence networks, and topic modeling.

---

## Repository contents

```
config/           # filters, scrape plan, seed dictionary (anchors), analysis knobs
data/
  processed/      # Phase 1 cleaned corpus CSVs
  analysis/       # Corrected analysis corpus CSVs (primary research tables)
scripts/          # scrape → filter → analysis pipeline
docs/             # methods notes and scraping ops tips
tests/            # unit tests for shared analysis helpers
run_analysis.py   # Phase 2–4 orchestration (PPMI default path)
run_contextual_phase2.py   # frozen BERT path + methods join
run_predictive_phase2.py   # Word2Vec / FastText evaluation
```

### Included data

| File | Rows (approx.) | Role |
|------|----------------|------|
| `data/analysis/corpus_a.csv` | 906 | **Primary** migration corpus for Phases 2–4 |
| `data/analysis/corpus_b.csv` | 687 | **Primary** grief corpus for Phases 2–4 |
| `data/processed/corpus_a.csv` | 323 | Earlier Phase 1 freeze (pre-filter recovery) |
| `data/processed/corpus_b.csv` | 687 | Phase 1 grief corpus |

Prefer **`data/analysis/`** for new analysis. The analysis corpus recovers posts that an over-broad keyword filter had dropped from Corpus A (substring false positives). Raw Apify dumps and large intermediates are not in git (regenerate or restore locally).

Typical columns: `post_id`, `corpus`, `subreddit`, `title`, `body`, `text`, `created_at`, `url`, `flair`, `score`, `word_count`, `username`, plus cleaned/token fields depending on stage.

> **Privacy:** CSVs include Reddit identifiers and post text. This repository is private by design. Do not publish row-level dumps without ethics/privacy review.

---

## Setup

### Scrape / Phase 1 environment

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Apify auth (pick one):

- `apify login` (CLI keyring), or  
- `$env:APIFY_TOKEN = "apify_api_..."` from [Apify Integrations](https://console.apify.com/settings/integrations)

### Analysis environment (Phases 2–4)

Uses a **separate** venv so the torch/transformers stack does not disturb Phase 1.

```powershell
# Use a Python 3.12 base that matches your machine
python -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -U pip
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements.txt -r requirements-analysis.txt -c constraints-analysis.txt
.\.venv-analysis\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis-torch.txt -c constraints-analysis.txt
.\.venv-analysis\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv-analysis\Scripts\python.exe -m spacy download en_core_web_md
.\.venv-analysis\Scripts\python.exe scripts\check_env.py
```

---

## Data collection (optional)

Scraping costs Apify credits. Pilot first:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/01_scrape_apify.py --mode pilot --corpus both --listings-only
```

Full collection (example — sequential is safest on limited concurrent memory):

```powershell
python scripts/01_scrape_apify.py --mode full --corpus both --engine trudax --searches-only --resume --workers 1 --timeout 3600
python scripts/03_filter_and_normalize.py
python scripts/04_preprocess_nlp.py
```

Config: `config/filters.yaml`, `config/scrape_runs.yaml`.  
Ops notes: `docs/SPEED.md`, `docs/HOME_RESUME.md`.

---

## Analysis pipeline

Seed dictionary: **4 domains / 71 anchors** in `config/anchors.yaml`  
(Temporal, Identity, Absence, Digital).

### Default path (PPMI + emotion + topics)

```powershell
.\.venv-analysis\Scripts\python.exe run_analysis.py
.\.venv-analysis\Scripts\python.exe -m pytest
```

### Frozen BERT (contextual) path

```powershell
.\.venv-analysis\Scripts\python.exe run_contextual_phase2.py
```

### Word2Vec / FastText (empirical small-corpus evaluation)

```powershell
.\.venv-analysis\Scripts\python.exe run_predictive_phase2.py
```

Results land under `outputs/tables/`, `outputs/figures/`, `outputs/summaries/` (local; not committed by default).  
Methods summary: `docs/METHODS_RESULTS.md`.

### Method overview

| Method | Role |
|--------|------|
| **PPMI** | Shared-basis context profiles; strongest A/B collocational contrast |
| **BERT** (`bert-base-uncased`) | Frozen contextual senses; stable neighbours at current N |
| **Word2Vec / FastText** | Trained on-site; second-order profiles stable, neighbour lists weak at current N |
| **Emotion** | DistilRoBERTa-emotion on ±5-token anchor windows (ABSA *proxy*) |
| **Topics** | Pooled NMF with bootstrap stability (selected k=8) |

Interpretation limits (current sample sizes ~906 / 687):

- Prefer **domain-level** claims; most individual anchors are rare.  
- Digital domain is fragile (polysemy / sparse tech terms).  
- Emotion is a window proxy, not a trained aspect model.  
- Only one NMF topic currently meets the high-stability “core” bar (study-abroad language).

---

## Tests

```powershell
.\.venv-analysis\Scripts\python.exe -m pytest
```

---

## License / ethics

Reddit content is subject to Reddit’s terms and user privacy expectations. Use for research only; do not redistribute personal data without appropriate review. Apify usage is billed to your own account.
