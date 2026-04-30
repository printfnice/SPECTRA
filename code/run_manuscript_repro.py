from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = Path(os.environ.get("SPECTRA_RESULTS_DIR", "/results"))
DEFAULT_LOCAL_RESULTS = ROOT / "results" / "exported_results"
SEED_OUTPUT_DIR = Path(os.environ.get("SPECTRA_SEED_OUTPUT_DIR", "/data/spectra_output_seed"))
LOCAL_SEED_TARBALLS = [
    ROOT / "data" / "seed_assets" / "spectra_output_seed.tar.gz",
    ROOT / "data" / "spectra_output_seed.tar.gz",
]
LOCAL_EXTRACTED_SEED_DIR = ROOT / "data" / "_staged_seed_output"
RUN_LOG = ROOT / "results" / "run_log.txt"

PLOT_SCRIPTS = [
    "pipeline/generate_tables.py",
    "pipeline/plot_sota_comparison.py",
    "pipeline/plot_supcon_ablation.py",
    "pipeline/plot_gate_weights.py",
    "pipeline/plot_pathway_scale.py",
    "pipeline/plot_gate_entropy.py",
    "pipeline/plot_tsne_umap.py",
    "pipeline/plot_unified_comparison.py",
    "pipeline/plot_calibration.py",
    "pipeline/plot_evi_fidelity.py",
    "pipeline/plot_efficiency_table.py",
    "pipeline/plot_gate_sample_distribution.py",
]

RESULT_GLOBS = [
    "output/figures/Fig1_custom_architecture.png",
    "output/figures/Fig2_sota_comparison.pdf",
    "output/figures/Fig2_sota_comparison.png",
    "output/figures/Fig3_supcon_ablation.pdf",
    "output/figures/Fig3_supcon_ablation.png",
    "output/figures/Fig4_gate_weights.pdf",
    "output/figures/Fig4_gate_weights.png",
    "output/figures/Fig5_pathway_scale.pdf",
    "output/figures/Fig5_pathway_scale.png",
    "output/figures/Fig6_gate_entropy.pdf",
    "output/figures/Fig6_gate_entropy.png",
    "output/figures/Fig7_tsne_umap.pdf",
    "output/figures/Fig7_tsne_umap.png",
    "output/figures/FigS_unified_comparison.pdf",
    "output/figures/FigS_unified_comparison.png",
    "output/figures/FigS_calibration_metrics.pdf",
    "output/figures/FigS_calibration_metrics.png",
    "output/figures/FigS_calibration_reliability.pdf",
    "output/figures/FigS_calibration_reliability.png",
    "output/figures/FigS_evi_fidelity.pdf",
    "output/figures/FigS_evi_fidelity.png",
    "output/figures/FigS_efficiency.pdf",
    "output/figures/FigS_efficiency.png",
    "output/figures/FigS_gate_sample_distribution.pdf",
    "output/figures/FigS_gate_sample_distribution.png",
    "output/figures/FigS_tsne_umap_velmeshev.pdf",
    "output/figures/FigS_tsne_umap_velmeshev.png",
    "output/tables/*.tex",
    "output/tables/*.md",
    "metadata/artifact_manifest.tsv",
]


def _ensure_results_dir() -> Path:
    if RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return RESULTS_DIR
    if str(RESULTS_DIR) != "/results":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return RESULTS_DIR
    DEFAULT_LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    return DEFAULT_LOCAL_RESULTS


def _reset_local_output() -> None:
    for subdir in [OUTPUT_DIR / "data", OUTPUT_DIR / "figures", OUTPUT_DIR / "tables"]:
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)


def _stage_static_inputs() -> None:
    ref_map = ROOT / "data" / "reference" / "cellchat_pathway_map.csv"
    target_map = ROOT / "data" / "cellchat_pathway_map.csv"
    target_map.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref_map, target_map)

    fig1 = ROOT / "assets" / "Fig1_custom_architecture.png"
    shutil.copy2(fig1, FIGURES_DIR / "Fig1_custom_architecture.png")


def _extract_local_seed_tarball(seed_tarball: Path) -> Path:
    if LOCAL_EXTRACTED_SEED_DIR.exists():
        shutil.rmtree(LOCAL_EXTRACTED_SEED_DIR)
    LOCAL_EXTRACTED_SEED_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(seed_tarball, "r:gz") as tar:
        tar.extractall(LOCAL_EXTRACTED_SEED_DIR)

    extracted_root = LOCAL_EXTRACTED_SEED_DIR / "spectra_output_seed"
    if extracted_root.exists():
        return extracted_root
    return LOCAL_EXTRACTED_SEED_DIR


def _resolve_seed_source() -> Path:
    if SEED_OUTPUT_DIR.exists():
        if SEED_OUTPUT_DIR.is_dir():
            return SEED_OUTPUT_DIR
        if SEED_OUTPUT_DIR.is_file() and SEED_OUTPUT_DIR.suffixes[-2:] == [".tar", ".gz"]:
            return _extract_local_seed_tarball(SEED_OUTPUT_DIR)

    for candidate in LOCAL_SEED_TARBALLS:
        if candidate.exists():
            return _extract_local_seed_tarball(candidate)

    raise FileNotFoundError(
        "Seed manuscript outputs were not found. "
        "Provide a directory via SPECTRA_SEED_OUTPUT_DIR or place "
        "spectra_output_seed.tar.gz under data/seed_assets/."
    )


def _stage_seed_outputs() -> None:
    seed_source = _resolve_seed_source()

    for item in seed_source.iterdir():
        if item.name in {"figures", "tables"}:
            continue
        dst = OUTPUT_DIR / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)


def _run_script(rel_path: str) -> None:
    script_path = CODE_DIR / rel_path
    cmd = [sys.executable, str(script_path)]
    with RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n=== RUN {rel_path} ===\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{rel_path} failed with code {proc.returncode}")


def _collect_results(results_dir: Path) -> None:
    for pattern in RESULT_GLOBS:
        for src in ROOT.glob(pattern):
            rel = src.relative_to(ROOT)
            dst = results_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    run_log_dst = results_dir / "run_log.txt"
    try:
        same_target = RUN_LOG.resolve() == run_log_dst.resolve()
    except FileNotFoundError:
        same_target = False
    if not same_target:
        shutil.copy2(RUN_LOG, run_log_dst)


def main() -> None:
    results_dir = _ensure_results_dir()
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("SPECTRA Code Ocean manuscript reproduction run\n", encoding="utf-8")

    _reset_local_output()
    _stage_seed_outputs()
    _stage_static_inputs()

    for rel_path in PLOT_SCRIPTS:
        _run_script(rel_path)

    _collect_results(results_dir)
    print(f"Results exported to: {results_dir}")


if __name__ == "__main__":
    main()
