# Phase 1 complete (2026-08-27 rebuild)

Final research sample after the sampling-frame corrections. No Apify was used
in this rebuild; collection is Arctic Shift plus already-downloaded raw dumps.

## Rules applied

- Posts only (no comments)
- Body longer than 100 words (title is not counted toward the threshold)
- Dates fail-closed: `2023-07-01 <= created_at < 2026-08-24`
- Corpus A keyword blacklist (whole-word) plus procedural flair drop
  (`visa`, `citizen`, `red tape`, `tax`, `financial`)
- Per-subreddit quotas, then preferred-flair first, remainder seeded-random
- Corpus A quotas proportional to the qualifying pool
- Corpus B scarce-first so `r/bereavement` is fully included

## Counts

| Corpus | Posts | Subreddits |
|--------|------:|------------|
| A | 10,000 | expats 3,889; StudyAbroad 3,240; immigration 2,871 |
| B | 10,000 | GriefSupport 9,752; bereavement 248 |

- Unique IDs within each corpus: 10,000
- Shared IDs A ∩ B: 0
- Empty / `[removed]` / `[deleted]` bodies: 0
- Body word count ≤ 100: 0
- Unparsable timestamps: 0
- Out-of-window timestamps: 0

Seed: `sampling_seed: 20260827` in `config/filters.yaml`.
