"""Phase 2 - PCA and t-SNE views of the eight corpus-domain profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import load_analysis_config  # noqa: E402
from analysis_io import banner, item, out_path, save_summary  # noqa: E402
from plotting import apply_style, fig_scatter, save_fig  # noqa: E402


def load_profiles(analysis_dir: Path, backend: str) -> tuple[np.ndarray, list[str], list[str]]:
    matrices = []
    names: list[str] | None = None
    labels: list[str] = []
    groups: list[str] = []
    for corpus in ("a", "b"):
        path = analysis_dir / "vectors" / f"domain_{corpus}_{backend}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/12_build_embeddings.py --unit domain first")
        with np.load(path, allow_pickle=True) as archive:
            current_names = [str(value) for value in archive["names"]]
            matrix = np.asarray(archive["matrix"], dtype=float)
        if names is not None and current_names != names:
            raise SystemExit("domain vector files have different target order")
        names = current_names
        matrices.append(matrix)
        labels.extend(f"{name} ({corpus.upper()})" for name in current_names)
        groups.extend([corpus] * len(current_names))
    return np.vstack(matrices), labels, groups


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA and t-SNE figures for domain profiles")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(seed=args.seed)
    backend = args.backend or str(cfg.get("vectors.backend"))
    matrix, labels, groups = load_profiles(cfg.path("analysis_dir"), backend)
    style = apply_style(cfg.section("figures"))
    corpus_labels = {key: str(value) for key, value in cfg.get("figures.corpus_labels").items()}
    colors = {
        corpus_labels[key]: str(value)
        for key, value in cfg.get("figures.corpus_colors").items()
    }
    display_groups = [corpus_labels[group] for group in groups]

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Standardisation prevents a handful of high-variance PPMI dimensions from
    # determining the projection solely by scale.
    scaled = StandardScaler().fit_transform(matrix)
    pca_model = PCA(n_components=2, random_state=cfg.seed)
    pca = pca_model.fit_transform(scaled)
    fig, _ = fig_scatter(
        pca[:, 0], pca[:, 1], labels, display_groups,
        style=style,
        title="Domain context profiles: PCA",
        xlabel=f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})",
        colors=colors,
    )
    # PPMI keeps default names for backward compatibility; contextual is tagged.
    if backend == "ppmi":
        pca_path = out_path("figures", "p2", 18, "domain_profiles_pca")
        tsne_path = out_path("figures", "p2", 18, "domain_profiles_tsne")
        summary_path = out_path("summaries", "p2", 18, "semantic_figures", ext="json")
    else:
        pca_path = out_path("figures", "p2", 18, "domain_profiles_pca", "contextual")
        tsne_path = out_path("figures", "p2", 18, "domain_profiles_tsne", "contextual")
        summary_path = out_path("summaries", "p2", 18, "semantic_figures", "contextual", ext="json")

    written = save_fig(fig, pca_path, formats=list(cfg.get("figures.formats")))

    perplexity = min(3.0, float(len(matrix) - 1))
    tsne = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=cfg.seed,
    ).fit_transform(scaled)
    fig, _ = fig_scatter(
        tsne[:, 0], tsne[:, 1], labels, display_groups,
        style=style,
        title="Domain context profiles: t-SNE (descriptive)",
        xlabel="t-SNE 1", ylabel="t-SNE 2", colors=colors,
    )
    written.extend(save_fig(fig, tsne_path, formats=list(cfg.get("figures.formats"))))

    summary = {
        "backend": backend,
        "n_profiles": len(matrix),
        "pca_explained_variance": pca_model.explained_variance_ratio_.tolist(),
        "tsne_perplexity": perplexity,
        "note": "t-SNE over eight domain-corpus profiles is descriptive and not inferential.",
        "figures": [str(path) for path in written],
    }
    save_summary(summary, summary_path)
    banner("Semantic figures")
    for path in written:
        item(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
