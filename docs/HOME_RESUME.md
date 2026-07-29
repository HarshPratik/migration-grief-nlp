# Resume scraping when you're home

## What went wrong (2026-07-22)

1. **Apify concurrent memory limit = 16384 MB**  
   Each `trudax` run requested **2048 MB**. About **8** concurrent runs fill the account.  
   Fire-and-forget + earlier parallel workers left the slot full → new starts failed with:

   > exceed the memory limit of 16384MB … currently used: 16384MB, requested: 2048MB

2. **CLI parser bug**  
   Some starts actually succeeded (`"ok": true` + `run.id`) but exit code was non-zero, so the script logged them as FAILED. **Fixed.**

3. **Local processes**  
   All laptop orchestrators were **stopped**. Nothing is running locally now.

## Your data is safe

- Local dumps under `data/raw/corpus_a` and `data/raw/corpus_b` are **kept**
- Good runs (≥10 posts) will still be **skipped** so you don't re-pay for ~750+ posts
- Thin 1-item timeout dumps are eligible to retry

## Before restarting

1. Open https://console.apify.com/actors/runs  
2. Let RUNNING jobs **finish**, or **abort** stuck ones to free memory  
3. Confirm free usage remaining under Billing  

## When you say "start it again"

### Option A — laptop sequential (most reliable on this plan)

```powershell
cd D:\research_paper
.\.venv\Scripts\Activate.ps1
python scripts/01_scrape_apify.py --mode full --corpus both --engine trudax --searches-only --resume --workers 1 --timeout 1200
```

Keep the laptop awake. One run at a time (~2 GB).

### Option B — fire-and-forget (close laptop after launch)

```powershell
cd D:\research_paper
.\.venv\Scripts\Activate.ps1
python scripts/05_fire_and_forget.py --dry-run
python scripts/05_fire_and_forget.py --min-items 10 --memory 1024 --stagger-seconds 90
```

Later download:

```powershell
python scripts/02_download_datasets.py
python scripts/03_filter_and_normalize.py
python scripts/04_preprocess_nlp.py
```

## Do not

- Use `--workers 3` on this plan (fills 16 GB immediately)
- Start 30+ fire-and-forget jobs while old runs still hold memory
