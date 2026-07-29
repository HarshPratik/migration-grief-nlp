"""Phases 2-4 - create a sanitized, integrity-checked lead-author bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item  # noqa: E402

SAFE_TABLE_PREFIXES = (
    "p2_09_", "p2_10_", "p2_13_", "p2_14_",
    "p3_15_", "p3_16_",
    "p4_17_topic_terms", "p4_17_topic_prevalence", "p4_17_k_sweep",
)
SAFE_SUMMARY_PREFIXES = (
    "p2_09_", "p2_10_", "p2_13_", "p2_14_", "p2_18_",
    "p3_15_", "p3_16_", "p3_19_",
    "p4_17_", "p4_20_",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sheet_name(path: Path, used: set[str]) -> str:
    base = path.stem.removeprefix("p2_").removeprefix("p3_").removeprefix("p4_")
    base = base.replace("__", "_")[:31]
    candidate = base
    suffix = 2
    while candidate in used:
        tag = f"_{suffix}"
        candidate = f"{base[:31-len(tag)]}{tag}"
        suffix += 1
    used.add(candidate)
    return candidate


def build_workbook(csv_files: list[Path], destination: Path, max_rows: int) -> list[str]:
    used: set[str] = set()
    included: list[str] = []
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        for path in csv_files:
            frame = pd.read_csv(path)
            if len(frame) > max_rows:
                continue
            frame.to_excel(writer, sheet_name=sheet_name(path, used), index=False)
            included.append(path.name)
        pd.DataFrame({"note": [
            "This bundle contains aggregate outputs only; no Reddit post IDs, usernames, excerpts, or timestamps.",
            "Corpus A is migration narrative; Corpus B is grief narrative.",
            "Digital dictionary results require manual concordance because polysemous anchors dominate.",
            "Only reproducible core topics should be interpreted as topic findings.",
            "Co-occurrence networks are exploratory because many high-PPMI edges have low counts.",
        ]}).to_excel(writer, sheet_name="READ_ME", index=False)
    return included


def selected_files(directory: Path, prefixes: tuple[str, ...], suffix: str) -> list[Path]:
    return sorted(
        path for path in directory.glob(f"*{suffix}")
        if path.name.startswith(prefixes)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create sanitized lead-author analysis bundle")
    parser.add_argument("--replace", action="store_true",
                        help="atomically replace an existing sanitized bundle")
    args = parser.parse_args()

    cfg = load_analysis_config()
    outputs = cfg.path("outputs_dir")
    bundle = outputs / "bundle_reviewed"
    archive = outputs / "research_analysis_bundle_reviewed.zip"
    staging = outputs / "bundle_reviewed.staging"
    archive_staging = outputs / "research_analysis_bundle_reviewed.staging.zip"
    existing = [path for path in (bundle, archive) if path.exists()]
    if existing and not args.replace:
        raise SystemExit(
            "sanitized output already exists; pass --replace to regenerate after preserving manual work"
        )
    if staging.exists() or archive_staging.exists():
        raise SystemExit("stale sanitized staging output exists; inspect it before retrying")
    staging.mkdir(parents=True)

    table_files = selected_files(outputs / "tables", SAFE_TABLE_PREFIXES, ".csv")
    topic_summary = json.loads(
        (outputs / "summaries" / "p4_17_topic_model.json").read_text(encoding="utf-8")
    )
    topic_tag = f"{topic_summary['method']}_k{int(topic_summary['k'])}"
    table_files = [
        path for path in table_files
        if not path.name.startswith(("p4_17_topic_terms", "p4_17_topic_prevalence"))
        or topic_tag in path.name
    ]
    summary_files = selected_files(outputs / "summaries", SAFE_SUMMARY_PREFIXES, ".json")
    workbook = staging / "research_analysis_tables.xlsx"
    included_sheets = build_workbook(
        table_files, workbook, max_rows=int(cfg.get("export.excel_max_rows"))
    )

    copy_groups = {
        "tables": table_files,
        "figures": sorted((outputs / "figures").glob("*.png")),
        "networks": sorted(
            path for path in (outputs / "networks").glob("*.graphml")
            if path.name in {"p3_16_cooccur__a.graphml", "p3_16_cooccur__b.graphml"}
        ),
        "summaries": summary_files,
        "config": sorted((ROOT / "config").glob("*.yaml")),
    }
    copied: list[Path] = [workbook]
    for directory, paths in copy_groups.items():
        target = staging / directory
        target.mkdir(parents=True, exist_ok=True)
        for source in paths:
            destination = target / source.name
            shutil.copy2(source, destination)
            copied.append(destination)

    readme = staging / "README.txt"
    readme.write_text(
        "Sanitized lead-author analysis bundle\n\n"
        "Contains aggregate CSV tables, figures, GraphML networks, summaries, configuration, "
        "and an Excel workbook. It contains no Reddit post IDs, usernames, excerpts, raw source "
        "filenames, or row-level timestamps. Restricted concordance, annotation, audit-ledger, "
        "and document-topic files remain local and are not included.\n",
        encoding="utf-8",
    )
    copied.append(readme)

    manifest = {
        "config_seed": cfg.seed,
        "privacy_class": "aggregate-only sanitized",
        "workbook_sheets_from": included_sheets,
        "files": [
            {
                "path": str(path.relative_to(staging)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(copied)
        ],
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(archive_staging, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, Path("bundle_reviewed") / path.relative_to(staging))

    # Build fully before touching the old sanitized deliverable. Replacement is
    # then two same-volume renames; failures cannot leave a partially built tree.
    if bundle.exists():
        backup = outputs / "bundle_reviewed.backup"
        if backup.exists():
            raise SystemExit(f"backup already exists: {backup}")
        bundle.rename(backup)
        try:
            staging.rename(bundle)
        except OSError:
            backup.rename(bundle)
            raise
        shutil.rmtree(backup)
    else:
        staging.rename(bundle)
    archive_staging.replace(archive)

    banner("Sanitized export bundle")
    item(f"workbook {workbook.name}")
    item(f"manifest outputs\\bundle_reviewed\\manifest.json ({len(manifest['files'])} files)")
    item(f"archive  {archive.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
