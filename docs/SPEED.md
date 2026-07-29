# Making Phase 1 scraping faster

## What actually limits speed

| Factor | Effect | Notes |
|--------|--------|--------|
| **Sequential actor runs** | One ~17 min run at a time | Biggest wall-clock bottleneck |
| **Reddit 403 / bot defense** | Retries, slow crawl | Why each post costs ~10–15s wall time |
| **Puppeteer (trudax)** | Browser-like, heavy | More reliable post body than bare HTTP sometimes |
| **Apify plan concurrency** | Caps parallel runs | Free/low plans often allow only 1–2 concurrent Actor runs |
| **Pay-per-result + proxy GB** | Cost, not pure speed | Residential proxy is slower/costlier but needed |

## Parallel runs (YES — do this)

```powershell
python scripts/01_scrape_apify.py --mode full --corpus both `
  --engine trudax --searches-only --resume --workers 3 --timeout 900
```

- **`--workers 3`** starts up to 3 Apify actor runs at once → ~3× wall-clock if the plan allows concurrency.
- If Apify queues extras, lower to `--workers 2`.
- Too high (e.g. 8+) can increase 403s and waste money with little gain.

## Proxies (already on)

We already pass **Apify RESIDENTIAL** proxies in actor input:

```yaml
proxy / proxyConfiguration:
  useApifyProxy: true
  apifyProxyGroups: [RESIDENTIAL]
```

| Proxy type | Speed | Reddit success |
|------------|-------|----------------|
| None / datacenter | Fast | Many **403** failures |
| **Residential (current)** | Slower | Much better success |
| Your home VPN | N/A for cloud actors | See below |

## VPN on your PC — does **not** help

Apify actors run on **Apify’s cloud**, not through your laptop network.

- Turning on a VPN locally **does not** change Reddit’s view of the scraper.
- Proxies must be configured **inside the actor input** (already done).
- VPN only matters for a **local** scraper (PRAW / custom script on your machine).

## Deduplication (YES — implemented)

1. **Live registry** `data/raw/seen_post_ids.json`  
   - Seeded from all existing dumps  
   - After each run: reports `raw` / `new` / `dup`  
2. **Filter step** `scripts/03_filter_and_normalize.py`  
   - `drop_duplicates(subset=["post_id"])` again before interim corpora  

Current pilot-scale dump had only **2 duplicate IDs** out of 170 rows (~1%).  
Overlapping search queries will raise dups; registry keeps counts honest.

## Other levers

| Lever | Speed gain | Tradeoff |
|-------|------------|----------|
| Smaller `maxItems` (e.g. 50–80) | Faster run finish | More actor starts |
| Larger `maxItems` (200+) | Fewer starts | Longer runs, more timeout risk |
| More diverse search queries | More unique posts | Some overlap |
| Skip listing-only thin URLs | Less wasted time | Already prefer `--searches-only` |
| Labrat parallel (25/run) | Fast small batches | Free-tier cap 25 |
| Higher memory on actor | Sometimes faster crawl | Slightly more $ |

## Recommended “fast path” for $52 balance

```powershell
# Stop any old sequential job first if needed, then:
python scripts/01_scrape_apify.py --mode full --corpus both `
  --engine trudax --searches-only --resume --workers 3 --timeout 900
```

Expected: near **2–3×** posts per wall-clock hour vs sequential, same $/post, until credits run out.
