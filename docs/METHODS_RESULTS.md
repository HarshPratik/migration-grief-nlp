# Methods and results on the current corpora (no new scraping)

**Last updated:** 2026-10-01 (human annotation validation §4.2 added)  
**Analysis corpora:** Corpus A (migration) **10,000** posts; Corpus B (grief) **10,000** posts (`data/analysis/`).  
**Seed dictionary:** 4 domains, **71** anchors (`config/anchors.yaml`, from `phases/Seed Dictionary.docx`).  
**Sampling:** Arctic Shift rebuild, seed `20260827`, window `2023-07-01 ≤ created_at < 2026-08-24`. See `data/processed/PHASE1_COMPLETE.md`.

This document records what was run against the phase briefs on the 10k/10k sample. Word2Vec/FastText full evaluation finished 2026-08-28.

---

## 1. Phase requirements → what we ran

| Phase doc | Requirement | Status on current data |
|-----------|-------------|------------------------|
| Phase 1 | Filter, clean, CSV A/B | **Done** — 10k/10k in `data/analysis`; audit in `data/processed/PHASE1_COMPLETE.md` |
| Phase 2 | Preprocess; embeddings; top-15 neighbors A/B; cosine A vs B; CSV/Excel; PCA/t-SNE | **Done** (PPMI, BERT, Word2Vec, FastText) |
| Phase 3 | Emotion on anchor windows; ABSA; stats; matrices; stacked bars/heatmaps | **Done** (window-proxy ABSA + DistilRoBERTa-emotion, **plus true ABSA** with DeBERTa-v3-absa — see §4.1) |
| Seed dict | Four domains | **Done** |
| Topics (pipeline Phase 4) | Topic model | **Done** (pooled NMF k=12, 11/12 core; independent LDA/NMF k=10,12,15; MiniLM k=12; **real BERTopic** k=14/14 — see §5.3) |
| New.docx (temporal) | User timelines; sadness slopes; BERTopic DTM; Year-1 vs Year-3+ betweenness | **Done** — BERTopic DTM now literal (`p5_46_*`); T0 = first sample post |

Word2Vec and FastText **are evaluated empirically** on these corpora (not dismissed without a run). They are trained from the research texts; reliability uses split-half **re-training** and second-order profiles over the shared basis (alignment-free). Frozen BERT still *measures* pretrained senses rather than estimating free parameters per type. See §2.4 and the multi-method scoreboard after `scripts/25_predictive_eval.py` / `scripts/23b_methods_dashboard.py`.

---

## 2. Representation methods (what we tried and why)

### 2.1 PPMI context profiles (count-based, alignment-free)

- Shared basis: types with frequency ≥ 20 in **both** corpora (**2,286** dimensions).
- For each anchor/domain in each corpus: window co-occurrence counts → PPMI → profile over the shared basis.
- Neighbours = top-15 basis dimensions by PPMI weight.
- **Primary unit:** domain (4/4 usable under reliability gate). Anchor usable count on 10k is in `p2_13_reliability__anchor.csv`; prefer domain-level claims.

**Outputs:** `data/analysis/vectors/*_ppmi.npz`, `outputs/tables/p2_13_*`, `p2_14_*` (no `__contextual` suffix).

### 2.2 Contextual BERT sense vectors (frozen `bert-base-uncased`)

- Encode each document once; mean of last 4 hidden layers; mean-pool wordpieces per whitespace token.
- Cache only tokens in anchors ∪ shared basis under `data/analysis/vectors/contextual_cache/`.
- Sense vector = mean of all cached occurrences of the target terms in that corpus.
- Neighbours = cosine of the sense vector to mean type embeddings of shared-basis words in that corpus.
- Split-half / permutation / bootstrap **never re-encode** — they re-average under document masks.

**Reliability note (important methodological result):** applying Mu & Viswanath all-but-the-top PCA on only **4** domain profiles *per half* destroyed domain reliability (cos≈0). Split-half reliability is measured **without** per-half ABTT; ABTT is only used when there are enough points (e.g. many type embeddings). After the fix, domain BERT reliability is **~0.96–0.999**.

**Outputs:** `*_contextual.npz`, `p2_13_*__contextual.csv`, `p2_14_*__contextual.csv`, long-form `p2_14_neighbors_top15__contextual.csv`.

### 2.3 Methods comparison package (PPMI vs BERT)

| File | Content |
|------|---------|
| `outputs/tables/p2_23_methods_reliability_compare.csv` | Side-by-side reliability |
| `outputs/tables/p2_23_methods_semantic_compare.csv` | Side-by-side cos / cos* / p |
| `outputs/tables/p2_23_methods_agreement.csv` | both_yes / both_no / disagree |
| `outputs/figures/p2_23_cosstar_ppmi_vs_contextual.png` | cos* scatter |
| `outputs/summaries/p2_23_methods_compare.json` | Counts |

**Significance agreement (usable FDR calls on 10k/10k):** **31** both_yes · **32** both_no · **12** disagree.

### 2.4 Word2Vec and FastText — empirical evaluation (not assumed a priori)

Trained on the analysis corpora with gensim 4.4 (numpy 2 compatible). Settings: skip-gram, dim=100, window=5, min_count=5, epochs=20, workers=1. Evaluation matches the other methods:

- Split-half **re-training** (12 splits)
- Second-order profiles over the **same 2,286-type shared basis** (alignment-free)
- Top-15 neighbours restricted to that basis
- Bootstrap (20) + stratified permutation (30) for A/B comparison

Runner: `python run_predictive_phase2.py` or `python scripts/25_predictive_eval.py`.

#### Domain reliability scoreboard (10k/10k)

| Method | Usable domains | Mean rel A / B | Mean neighbour Jaccard A / B | Chance Jaccard |
|--------|----------------|----------------|------------------------------|----------------|
| **BERT (contextual)** | 4/4 | **0.999 / 0.999** | **0.604 / 0.746** | 0.0033 |
| FastText | 4/4 | 0.992 / 0.990 | 0.171 / 0.360 | 0.0033 |
| Word2Vec | 4/4 | 0.988 / 0.986 | 0.130 / 0.287 | 0.0033 |
| PPMI | 4/4 | 0.548 / 0.577 | 0.042 / 0.053 | 0.0033 |

#### Domain cos* (A vs B similarity; higher = more similar)

| Domain | PPMI | BERT | Word2Vec | FastText |
|--------|-----:|-----:|---------:|---------:|
| Digital | 0.71 | 0.97 | 0.96 | 0.97 |
| Temporal | 0.71 | 0.92 | 0.98 | 0.98 |
| Absence | 0.67 | 0.93 | 0.97 | 0.98 |
| Identity | 0.70 | 0.86 | 0.96 | 0.97 |

#### What this means for the paper (honest reading)

1. **We did not skip Word2Vec/FastText.** They were trained and scored with the same reliability protocol.
2. **Second-order profiles are stable** for domain-pooled W2V/FT at this N (rel ≈ 0.99). So it is **false** to say “predictive embeddings produce pure noise here.”
3. **Neighbour lists are the weak part for W2V/FT:** mean Jaccard ~0.13–0.36 vs chance 0.0033 — above chance, and **far below BERT** (~0.60 / 0.75). Phase 2’s requested “top-15 nearest neighbours” is therefore **not the primary finding** under W2V/FT.
4. **cos* under W2V/FT is near 1.0** for all domains — little A/B contrast. **PPMI** shows the strongest collocational divergence; **BERT** is intermediate.
5. **FastText caveat:** subword vectors give high reliability even for near-zero frequency anchors; we still require `min_f ≥ 20` so those do not enter the FDR family.
6. **The 10k re-run is done** (`scripts/25_predictive_eval.py --algorithms word2vec fasttext --unit both --train-workers 8`). Neighbour Jaccard rose vs the 906/687 sample but remains far below BERT.

Multi-method tables:

| File | Content |
|------|---------|
| `outputs/tables/p2_23_reliability_scoreboard_domain.csv` | Domain scoreboard |
| `outputs/tables/p2_23_domain_min_reliability_by_method.csv` | Wide min-reliability |
| `outputs/tables/p2_23_domain_cosstar_by_method.csv` | Wide cos* |
| `outputs/tables/p2_23_all_methods_reliability.csv` | Stacked all backends |
| `outputs/tables/p2_23_all_methods_semantic.csv` | Stacked all backends |

---

## 3. Phase 2 results

### 3.1 Domain-level A vs B similarity

| Domain | PPMI cos* | PPMI sig (FDR) | BERT cos* | BERT sig (FDR) |
|--------|----------:|:--------------:|----------:|:--------------:|
| Digital | 0.71 | **Yes** (q=0.0001) | **0.97** | Yes (q=0.002) |
| Temporal | 0.71 | **Yes** (q=0.0001) | **0.92** | Yes (q=0.002) |
| Absence | 0.67 | **Yes** (q=0.0001) | **0.93** | Yes (q=0.002) |
| Identity | 0.70 | **Yes** (q=0.0001) | **0.86** | Yes (q=0.002) |

**How to read this:**

- **PPMI** asks: “Do the *local collocates* of these words look the same in A and B?” Answer: **no for all four domains** after FDR (q=0.0001). Digital is no longer the exception it was on the 906/687 sample.
- **BERT** asks: “Do the *contextual embeddings* of these words land in the same region of pretrained space?” Answer: **mostly yes** (very high cos*), but still **slightly lower** than a label-permutation null, so residual A/B structure remains.

These are complementary, not contradictory. For a paper: report both; do not claim Word2Vec-style “semantic change” without the reliability gate.

### 3.2 Declared domain-pair hypotheses

| Comparison | PPMI Δ(A−B) CI excludes 0? | BERT Δ(A−B) CI excludes 0? |
|------------|:--------------------------:|:--------------------------:|
| digital vs absence | No (diff +0.024) | **Yes** (diff +0.024, CI +0.014 to +0.034) |
| temporal vs absence | Yes (diff +0.081) | **Yes** (diff +0.100, CI +0.093 to +0.106) |
| identity vs absence | No (diff +0.003) | **Yes** (diff +0.021, CI +0.011 to +0.031) |
| digital vs temporal | Yes (diff −0.117) | **Yes** (diff +0.038, CI +0.028 to +0.049) |

PPMI pair CIs: `p2_14_domain_pair_similarity.csv`. BERT pair CIs: `p2_14_domain_pair_similarity__contextual.csv` (refreshed 2026-08-28 `--unit domain`). Supported/rejected calls: `p2_14_hypothesis_outcomes.csv`. BERT finds all four pair differences; PPMI is more conservative on digital/identity vs absence. The declared *temporal vs absence* story (closer in grief) is **not** supported: both methods put temporal–absence closer in **migration** (positive A−B). BERT *digital vs temporal* is also the opposite sign from PPMI.

### 3.3 Per-seed top-15 neighbours

Long-form exports (backend, unit, target, corpus, rank, neighbor):

- `outputs/tables/p2_14_neighbors_top15__ppmi.csv` — **2250** rows  
- `outputs/tables/p2_14_neighbors_top15__contextual.csv` — **2250** rows  

Qualitatively, BERT neighbours are more thematically coherent (e.g. absence → *lonely, lose, depressed, grief*). PPMI neighbours are corpus-specific collocates and often look less “semantic” but are the honest second-order signal.

### 3.4 Usable anchors (frequency floor + reliability)

| Backend | Domain usable | Anchor usable (min rel ≥ 0.25 and min f ≥ 20) |
|---------|---------------|-----------------------------------------------|
| PPMI | 4/4 | 29/71 |
| BERT | 4/4 | 40/71 |
| Word2Vec | 4/4 | 48/71 |
| FastText | 4/4 | 48/71 |

BERT often has **higher raw reliability** but the **same frequency floor** still excludes rare Digital/Identity terms that barely appear in one corpus.

### 3.5 Figures

| Method | PCA / t-SNE |
|--------|-------------|
| PPMI | `p2_18_domain_profiles_pca.png`, `…_tsne.png` |
| BERT | `p2_18_domain_profiles_pca__contextual.png`, `…_tsne__contextual.png` |

---

## 4. Phase 3 results (emotion / co-occurrence)

Already complete on the analysis corpora; not re-scraped.

| Item | Result |
|------|--------|
| Model | `j-hartmann/emotion-english-distilroberta-base` (7 classes) |
| Design | ±5 token windows; author-capped; polysemy-strict for inference; BH-FDR |
| Domain Identity | **q = 0.0003** different A vs B (sadness A−B = −0.17) |
| Domain Absence | **q = 0.0003** |
| Domain Temporal | **q = 0.0003** |
| Domain Digital | **q = 0.0003** (neutral A−B = +0.21) |
| ABSA | Window **proxy** (emotion of local context; aspect = anchor) — superseded/complemented by §4.1 true ABSA |
| Networks | PPMI co-occurrence, top 300 edges, GraphML + plots |
| Annotation sample | 150 windows **coded by human** and scored — see §4.2 |

Figures: `p3_19_emotion_stacked_domain.png`, `p3_19_emotion_difference_heatmap.png`, `p3_20_cooccurrence__{a,b}.png`.

### 4.1 True ABSA (added 2026-09-11; closes the "not a trained ABSA model" gap)

| Item | Result |
|------|--------|
| Script | `scripts/45_absa_true.py` (+ `45b_absa_figures.py`) |
| Model | `yangheng/deberta-v3-base-absa-v1.1` — anchor passed as an **explicit aspect** (`[CLS] sentence [SEP] anchor [SEP]`) on the **raw** sentence, not the lemmatized window |
| Unit | one (post, anchor) pair, first mention; 8,911 pairs (A) / 29,679 (B) |
| Stats | identical protocol to §4: author cap, polysemy exclusion, post-level aggregation, chi-square/MC, JS + stratified permutation, BH-FDR |
| Identity | **JS 0.219, q = 0.0004** — grief more aspect-negative (0.61 vs 0.42), migration more aspect-neutral |
| Absence | **JS 0.104, q = 0.0004** — migration more aspect-negative (0.49 vs 0.38) |
| Temporal | **JS 0.092, q = 0.0004** — migration more positive (0.15 vs 0.09) |
| Digital | **JS 0.040, q = 0.486 — NOT different.** The §4 window-emotion Digital effect (neutral +0.21 in migration) does **not** survive as aspect-directed polarity: it was emotion of the surrounding context, not attitude toward the digital tool itself |

Tables: `p3_45_absa_comparison.csv`, `p3_45_absa_distribution_matrix.csv`. Figures: `p3_45_absa_stacked_domain__{a,b}.png`, `p3_45_absa_difference_heatmap.png`. Scored pairs: `data/analysis/absa_{a,b}_deberta-v3-base-absa-v1_1.parquet`.

### 4.2 Human validation of the window-emotion labels (added 2026-10-01)

The 150-window sample (`p3_22`, corpus × domain stratified, non-polysemous anchors preferred) was coded blind: category (`aspect_owned` / `surrounding_context` / `ambiguous` / `wrong_anchor_match`), dominant emotion, confidence 1–3. Model labels stayed hidden until coding was complete. Script: `scripts/47_annotation_agreement.py`.

| Human category | n | share |
|----------------|--:|------:|
| aspect_owned | 31 | 20.7% |
| surrounding_context | 50 | 33.3% |
| ambiguous | 15 | 10.0% |
| wrong_anchor_match | 54 | 36.0% |

| Subset | n | accuracy | Cohen's κ |
|--------|--:|---------:|----------:|
| aspect_owned only | 31 | 0.419 | 0.097 |
| aspect_owned + surrounding_context (primary) | 81 | 0.444 | 0.229 |
| all coded rows | 150 | 0.460 | 0.217 |

| Auxiliary check | Result |
|-----------------|--------|
| Model `p_max`, agrees vs disagrees (primary subset) | 0.855 vs 0.647, Mann–Whitney p = 3.4e-6 |
| Coder confidence 3 rows | 68.2% agreement |
| Coder confidence 1–2 rows | ~35% agreement |
| Human emotion mix on aspect_owned | sadness 24 / neutral 4 / anger 2 / joy 1 (77% sadness) |

**Polysemy-flag audit.** All 150 sampled windows were on anchors flagged `polysemous: false`, yet the coder still rejected 54 as wrong-sense matches (`internet` 12, `schedule` 7, `weekend` 5, `alone`/`forgotten`/`distance`/`whatsapp` 4 each). The flag under-detects surface-sense drift — e.g. *distance* in "long distance relationship", *internet* as infrastructure rather than contact medium. Concentrated in Corpus A (38/54; `a/digital` is 79% wrong-sense), rare in grief absence/identity.

**Interpretation for the write-up.** (i) Window-emotion labels are a *context* proxy: only ~1/5 of scored windows carry emotion about the seed term, so §4 numbers describe the neighbourhood's affect, not the anchor's. (ii) Per-window labels are too noisy to quote individually; domain-level distributions remain defensible because confidence is calibrated and errors are not obviously corpus-skewed in the same direction. (iii) This sharpens §4.1: the true-ABSA Digital null is unsurprising when half of Digital's sampled windows are wrong-sense matches — aspect-directed measurement was needed, and it is the instrument that should carry aspect-level claims.

Tables: `p3_47_annotation_{agreement,categories,confusion,by_stratum,by_confidence,wrong_anchors,coding}.csv`. Figures: `p3_47_annotation_categories.png`, `p3_47_annotation_confusion.png`. Summary: `outputs/summaries/p3_47_annotation_agreement.json`. All row-level outputs stay local (contain post excerpts).

---

## 5. Phase 4 results (topics)

### 5.1 Pooled NMF (stability-selected)

| Item | Result |
|------|--------|
| Method | Pooled NMF, TF-IDF, stability selection |
| Selected k | **12** (k=8 has higher mean agreement 0.96; k=12 keeps 11/12 core and higher agreement-above-null) |
| Core topics (recurrence ≥ 0.8) | **11/12** — A: university, study-abroad, language/culture, passport, move/job; B: father loss, cancer, grief support, love-miss, family deaths, hospital days; shared: friends |
| Other topics | Topic 1 (language/culture) is the only non-core at k=12 (recurrence 0.5) |

Figures: `p4_20_topic_prevalence.png`, `p4_20_topic_stability.png`.

### 5.2 Independent models (per corpus)

The Phase 4 brief also asks for **independent** topic models on A and B (not a pooled fit), 10–15 topics, unforced Digital–Absence clustering, keyword co-occurrence, interactive views, and Identity–Time–Digital bridges.

| Item | Result |
|------|--------|
| Script | `scripts/33_phase4_full_compare.py` |
| LDA / NMF | Fitted **separately** on A and on B at **k = 10, 12, 15** |
| MiniLM | `all-MiniLM-L6-v2` embeddings + KMeans + class-based TF-IDF at **k = 12** (CUDA) |
| Digital–Absence mix at k=12 | **0** topics for LDA, NMF, and MiniLM on both corpora |
| Digital in top-10 | Almost never; Digital is too sparse to surface as its own topic |
| Absence | Appears in Corpus B NMF (grief/loss/family), not mixed with Digital |
| Networks | LDA k=12 topic-keyword GraphML + PNG; Identity / Time / Digital / Absence bridges in `p4_33_domain_bridges.csv` |
| Interactive | `docs/figures/p4_33_interactive__{lda,nmf,minilm}__{a,b}.html` |

Independent models do **not** force Digital and Absence into one cluster. That is the honest result of leaving the clustering unforced.

Figures: `p4_33_topic_network__{a,b}.png`. Tables regenerate under `outputs/tables/p4_33_*`.

### 5.3 Real BERTopic (added 2026-09-11; replaces the MiniLM substitution)

The earlier ban was environmental (UMAP/HDBSCAN dependency conflicts), not methodological. BERTopic 0.17.4 now runs in an **isolated venv** (`.venv-bertopic`, CPU torch; `.venv-analysis` untouched and still passing all tests). Embeddings are precomputed under `.venv-analysis` on CUDA (`scripts/46_bertopic_embed.py` → `data/analysis/vectors/minilm_full_{a,b}.npy`), then clustered by `scripts/46_bertopic_run.py`.

| Item | Corpus A | Corpus B |
|------|----------|----------|
| Initial HDBSCAN clusters (min_cluster_size=10, EOM) | 121 | 60 |
| Topics after `nr_topics=15` reduction | 14 | 14 |
| Raw outliers | 48.6% | 37.4% |
| Outliers after `reduce_outliers(strategy="embeddings")` | 0 | 0 |
| Digital–Absence unforced mix (top-10 words) | **0 topics** | **0 topics** |

Corpus A topics: general expat talk, immigration/ICE, passport & citizenship documents, health insurance, banking/phone/SIM, OPT/employer sponsorship, university admissions, international students, family sponsorship, food, study tools, Hong Kong, car import; plus one multilingual cluster. Corpus B topics: mother-loss, father-loss, pet loss, love/missing, holidays (Christmas/birthday), family death, pregnancy/baby loss, grief support, visitation dreams, belongings, funeral/cremation, eating/appetite, gifts/anniversaries; plus one multilingual cluster. Both corroborate the NMF k=12 structure.

Tables: `p4_46_bertopic_topics__{a,b}.csv` (top-10 c-TF-IDF words + weights), `p4_46_bertopic_doc_topics__{a,b}.csv`, `p4_46_digital_absence_mix__bertopic.csv`. Interactive: `p4_46_interactive_topics__bertopic__{a,b}.html`, `p4_46_barchart__bertopic__{a,b}.html`. Dynamic topics (New.docx Task 3, now literal BERTopic): `p5_46_bertopic_topics_over_time__{a,b}.csv` + `p5_46_topics_over_time__bertopic__{a,b}.html`.

---

## 6. What works vs what does not (for the write-up)

### Works well on this sample size

1. **Domain-level** analysis (4 seed domains) under both PPMI and BERT.  
2. **Frozen BERT** sense vectors with document-level caching (stable split-half reliability).  
3. **PPMI** collocational contrast (clearer A/B *divergence* in cos*).  
4. **Emotion window proxy** with author clustering + FDR (all four domains, q=0.0003) — now human-validated as a *domain-distribution* instrument only (§4.2).  
5. **NMF with stability** (honest: 11/12 core at k=12). Independent LDA/NMF/MiniLM agree that Digital–Absence does not cluster together when unforced.  
6. **Reliability gates** — prevents publishing neighbour lists as findings when they are chance-level.
7. **Honest temporal nulls** — calendar sadness stationary in both corpora; Digital–Absence still never mixes; hybrid betweenness almost absent.

### Does not work / weak

1. **Word2Vec / FastText neighbour lists** — above chance but far below BERT; do not treat W2V/FT top-15 as the primary finding.  
2. **Most individual anchors** — still rare in one corpus; prefer domain-level claims.  
3. **Digital domain** remains the sparsest (~8% of posts) even though emotion and PPMI now clear FDR.  
4. **Topic 1** (language/culture) is the only non-core topic at k=12.  
5. **True ABSA** — now run (§4.1); its Digital null qualifies the §4 window-emotion Digital finding.  
6. **Declared pair hypotheses vs 10k CIs** — BERT finds all four pair differences, but *temporal vs absence* is closer in migration, not grief; BERT and PPMI disagree on the sign of *digital vs temporal*.
7. **Author-relative T0** — 95% singleton authors; cannot recover the origin event. Calendar time is the usable clock.
8. **BERTopic granularity/outlier trade-off** — HDBSCAN `min_cluster_size=10` yields 60–121 raw clusters but 37–49% outliers (reassigned via embedding-space nearest topic); larger `min_cluster_size` collapses to 4–7 mega-topics. The 14-topic reduction is a comparability choice, not a data-driven optimum.
9. **Per-window emotion labels** — human audit (§4.2): only 20.7% of scored windows are aspect-owned, 36% are wrong-sense matches on *non-polysemous* anchors, and human–model label agreement is κ = 0.23 on usable windows. The polysemy flag does not catch this drift; domain distributions survive, single-window claims do not.

### Methodological lesson (BERT ABTT)

All-but-the-top anisotropy correction is unsafe when the profile set is tiny (4 domains). Measuring reliability on raw mean-pooled senses, and restricting ABTT to large type sets, is required for honest numbers.

---

## 7. How to reproduce

### Full Phase 2–4 (PPMI default chain)

```powershell
cd D:\research_paper
.\.venv-analysis\Scripts\python.exe run_analysis.py
```

### BERT Phase 2 + methods compare (after PPMI tables exist)

```powershell
.\.venv-analysis\Scripts\python.exe run_contextual_phase2.py
# or from reliability if cache already built:
.\.venv-analysis\Scripts\python.exe run_contextual_phase2.py --from-step reliability
```

### Independent Phase 4 topic comparison

```powershell
.\.venv-analysis\Scripts\python.exe scripts\33_phase4_full_compare.py
```

### True ABSA (Phase 3 addendum)

```powershell
.\.venv-analysis\Scripts\python.exe scripts\45_absa_true.py     # caches per-corpus scores, then stats
.\.venv-analysis\Scripts\python.exe scripts\45b_absa_figures.py
```

### Real BERTopic (Phase 4 addendum)

```powershell
.\.venv-analysis\Scripts\python.exe scripts\46_bertopic_embed.py   # GPU MiniLM embeddings
.\.venv-bertopic\Scripts\python.exe scripts\46_bertopic_run.py     # UMAP+HDBSCAN+c-TF-IDF + DTM
```

### Annotation agreement (Phase 3 validation)

```powershell
# after the coded workbook/CSV is back in annotation/
.\.venv-analysis\Scripts\python.exe scripts\47_annotation_agreement.py
```

### Phase 5 temporal follow-up (New.docx)

```powershell
.\.venv-analysis\Scripts\python.exe scripts\40_temporal_stratify.py
.\.venv-analysis\Scripts\python.exe scripts\41_emotion_over_time.py
.\.venv-analysis\Scripts\python.exe scripts\42_topics_over_time.py --skip-bertopic
.\.venv-analysis\Scripts\python.exe scripts\43_temporal_graphs.py
.\.venv-analysis\Scripts\python.exe scripts\44_temporal_figures.py
```

### Neighbour long-form from comparison CSVs

```powershell
.\.venv-analysis\Scripts\python.exe scripts\14b_export_neighbors.py
```

### Shareable archive

Only **`outputs/research_analysis_bundle_reviewed.zip`** is approved for sharing (aggregate-only). Rebuild with `scripts/21_export_bundle.py --replace` after regenerating tables if you want BERT CSVs inside a new bundle (current reviewed zip may predate BERT exports — keep BERT tables under `outputs/tables/` for the paper appendix either way).

---

## 8. Phase 5 / New.docx (temporal follow-up)

Added 2026-09-01. Scripts `40`–`44`. T0 is **first post per username in this sample**, not the true migration or loss date.

### 8.1 Stratification

| | Corpus A | Corpus B |
|--|--------:|--------:|
| Authors | 9,375 | 9,107 |
| Repeat authors | 461 | 582 |
| Posts from repeat authors | 10.9% | 14.8% |
| Author-relative Year 1 / Year 3+ | 9,952 / **5** | 9,900 / **14** |
| Calendar Year 1 / Year 3+ (from 2023-07-01) | 2,686 / 3,918 | 2,767 / 3,706 |

`time_interval` = quarter since T0. Calendar quarters are the sensitivity clock. Corpus B remains grief/bereavement, not a “homeland network.”

### 8.2 Sadness decay vs stationarity

Same model as Phase 3: `j-hartmann/emotion-english-distilroberta-base`. Primary family = window sadness, all posts, bin-mean slopes, both clocks, BH-FDR.

- Calendar, migration: slope −0.0016, p = 0.28, q = 0.85 → **stationary/null** (consistent with the migrant-stationarity hypothesis).
- Calendar, grief: slope +0.0005, p = 0.71, q = 0.86 → **stationary/null** (**rejects** the predicted decay).
- Extras that agree: Theil–Sen, 2,000 permutations on bin means, full-post document emotion, author-clustered bootstrap on post-level OLS, repeat-author subsample, four seed domains.

Author-relative bin means are underpowered (migration: 2 bins). Do not read T0 slopes as event-time trajectories.

### 8.3 Dynamic topics

BERTopic now runs literally (§5.3, `.venv-bertopic`): `p5_46_bertopic_topics_over_time__{a,b}.csv` + interactive HTML. The earlier canonical DTM remains as a comparable extra: `all-MiniLM-L6-v2` embeddings + MiniBatchKMeans k=12 (global topic identity) + quarter-wise prevalence, and NMF k=12 independently per calendar year. Digital–Absence unforced mix = **0** in all of them.

### 8.4 Temporal graphs

PPMI ±5, top 300 edges, Year 1 vs Year 3+. Betweenness spike if late/early ≥ 2. English stoplist plus multilingual closed-class filter (late calendar slices otherwise collapse onto `que` / `ich` / `una`). Hybrid = spike **and** (seed-domain node or neighbours in two seed domains). Result: **one** hybrid, Temporal seed `miles` on migration calendar time, Δ ≈ 10⁻⁴. Not evidence of *Aufhebung*.

Outputs: `outputs/tables/p5_40_*`, `p5_41_*`, `p5_42_*`, `p5_43_*`; figures `docs/figures/p5_44_*`.

---

## 9. File map (main deliverables)

```
config/anchors.yaml, analysis.yaml, filters.yaml
data/analysis/corpus_{a,b}.parquet
data/analysis/vectors/*_ppmi.npz
data/analysis/vectors/*_contextual.npz
data/analysis/vectors/contextual_cache/
outputs/tables/p2_09_* … p2_14_* … p2_23_*
outputs/tables/p3_11_* p3_15_* p3_16_* p3_22_* p3_45_*
outputs/tables/p4_17_* p4_33_* p4_46_* p5_46_*
outputs/figures/p2_18_* p2_23_* p3_19_* p3_45_* p3_20_* p4_20_* p4_33_* p4_46_* p5_46_*
outputs/networks/*.graphml
scripts/33_phase4_full_compare.py
scripts/45_absa_true.py, scripts/45b_absa_figures.py
scripts/46_bertopic_embed.py, scripts/46_bertopic_run.py   (runs under .venv-bertopic)
docs/METHODS_RESULTS.md   ← this file
run_analysis.py
run_contextual_phase2.py
```

---

## 10. Bottom line for hypotheses

On the **10k/10k** sample (rebuild 2026-08-27; no further scraping needed):

- **PPMI:** all four seed domains **diverge** between migration and grief after FDR (q = 0.0001). Digital is no longer the borderline case it was on the 906/687 sample.  
- **BERT:** the same domains are **highly similar** across corpora (cos\* 0.86–0.97) but still not interchangeable under a permutation null (q = 0.002).  
- **Declared domain-pair hypotheses** mostly **fail or split by method.** Both PPMI and BERT put *temporal vs absence* closer in **migration**, opposite the brief. BERT and PPMI **disagree on the sign** of *digital vs temporal*. See `outputs/tables/p2_14_hypothesis_outcomes.csv`.  
- **Emotion** around anchors differs for **all four** domains (q = 0.0003), including Digital (migration windows more neutral) — but see the next bullet.  
- **True ABSA** (DeBERTa-v3, aspect = anchor): identity, absence and temporal polarity differ (q = 0.0004); **Digital does not** (q = 0.486) — the window-emotion Digital effect is context emotion, not attitude toward the tool.  
- **Topics:** pooled NMF **k = 12** keeps **11/12** core themes (A: university / study-abroad / passport / move-job; B: family death / cancer / grief support; shared: friends). k = 8 is more stable if you want fewer, cleaner cores. Independent LDA/NMF (k=10,12,15), MiniLM k=12, **and real BERTopic (14/14 topics)** all find **no Digital–Absence mixed topic** on either corpus.  
- **Time (New.docx):** sadness does **not** decay in grief and is **not** a moving baseline in migration on the calendar clock. Author-relative T0 cannot stand in for the origin event. Hybrid betweenness nodes are essentially absent. Digital–Absence still never mixes.  
- Report **PPMI and BERT together**. Word2Vec/FastText were run: domain profiles are stable, neighbour lists are the weak finding — do not headline them.

Digital coverage is still sparse (~8% of posts; **~2.6%** if polysemous anchors such as *connection* are dropped). Prefer **domain-level** claims; most individual seeds remain rare.
