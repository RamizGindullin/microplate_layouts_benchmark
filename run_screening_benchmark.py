#!/usr/bin/env python3
"""
run_screening_benchmark.py

Scriptified version of the screening benchmark pipeline that was previously
implemented via Jupyter notebooks (screening-experiments.ipynb and
screening-supplement.ipynb).

Goals:
- Reproduce the screening CSVs and figures consumed by the LaTeX supplement
  and 0b_figures_tables.tex with minimal disruption to filenames and paths.
- Keep all existing code under libraries/ intact and imported as a dependency.

This script assumes it is run from the evaluation_aaai26 directory (i.e., the
directory that contains libraries/, layouts/, generated-data/, etc.).
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

import libraries.screening as sc
import libraries.disturbances as dt
import libraries.normalization as nrm
import libraries.utilities as util


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScreeningConfig:
    # Core parameters (matching the main-paper / supplement screening setup)
    neg_controls: int = 10
    pos_controls: int = 10

    # Control distributions (from screening-experiments.ipynb)
    neg_control_mean: float = 100.0
    pos_control_mean: float = 40.0
    neg_stdev: float = 10.0
    pos_stdev: float = 10.0

    # Hit rates (1, 5, 10, 20, 30, 40%) and corresponding percent_non_active
    hit_rates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20, 0.30, 0.40])

    # Bowl-effect strengths used for the ROC/PR / expected-vs-obtained plots
    bowl_strengths_expected_obtained: List[float] = field(default_factory=lambda: [0.03, 0.06, 0.08])
    bowl_strength_strong: float = 0.2  # strong bowl-shaped effect (for ROC/PR grid)

    # Bowl-effect strengths for SSMD/Z' robustness metrics
    metrics_bowl_strengths: List[float] = field(
        default_factory=lambda: [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    )

    # Where to write data and plots (relative to evaluation_aaai26/)
    data_dir: Path = Path("generated-data/screening")
    figures_dir: Path = Path("generated-plots/screening-supplement")

    # Filename “tags” used in the original notebooks, kept fixed to avoid
    # breaking LaTeX references.
    residuals_run_tag: str = "20250623-ROC-supplement"
    metrics_run_tag: str = "20250623-reviewing"

    # How many plates per layout / condition (taken from the original setup)
    plates_per_layout: int = 40

    # Number of batches in the ROC/PR utilities
    batches_for_curves: int = 10

    # Paths to layout directories and patterns for 10–10 screening layouts
    # (Random, PLAID, COMPD). These mirror how the notebooks iterate layouts.
    plate_types: List[Dict] = field(default_factory=lambda: [
        {
            "type": "random",
            "dir": "layouts/screening_RANDM_layouts/",
            "regex": "plate_layout_rand_10-10_.*.npy",
        },
        {
            "type": "plaid",
            "dir": "layouts/screening_PLAID_layouts/",
            "regex": "plate_layout_10-10_.*.npy",
        },
        {
            "type": "compd",
            "dir": "layouts/screening_COMPD_layouts/",
            "regex": "plate_layout_10-10_.*.npy",
        },
    ])

    @property
    def root(self) -> Path:
        # Assumes the script is run from evaluation_aaai26.
        return Path.cwd()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_directories(cfg: ScreeningConfig) -> None:
    (cfg.root / cfg.data_dir).mkdir(parents=True, exist_ok=True)
    (cfg.root / cfg.figures_dir).mkdir(parents=True, exist_ok=True)


def iter_layout_files(cfg: ScreeningConfig):
    """
    Yield (plate_type_dict, layout_file_name) over all screening layouts for
    Random / PLAID / COMPD with 10 negative and 10 positive controls.

    plate_type_dict has keys: type, dir, regex.
    """
    for plate_type in cfg.plate_types:
        layout_dir = cfg.root / plate_type["dir"]
        if not layout_dir.exists():
            continue
        for fname in os.listdir(layout_dir):
            if not fname.endswith(".npy"):
                continue
            if not Path(fname).match(plate_type["regex"]):
                continue
            yield plate_type, fname


# ---------------------------------------------------------------------------
# Stage 1: Simulations → residuals CSVs + scores CSVs
# ---------------------------------------------------------------------------

def simulate_condition(
    cfg: ScreeningConfig,
    error_strength: float,
    hit_rate: float,
) -> None:
    """
    Simulate all layouts for a single (error_strength, hit_rate) condition and
    write both scores and residuals CSVs using the filenames expected by the
    LaTeX sources and utilities.plot_* functions.
    """
    data_dir = cfg.root / cfg.data_dir

    pna = 1.0 - hit_rate  # percent_non_active
    # Names follow the pattern used in the original notebooks and artifact map.
    scores_filename = (
        f"screening_scores_data-"
        f"{cfg.pos_controls}-{cfg.neg_controls}-"
        f"{error_strength}-pna-{pna}-{cfg.residuals_run_tag}.csv"
    )
    residuals_filename = (
        f"screening-residuals-"
        f"{cfg.pos_controls}-{cfg.neg_controls}-"
        f"{error_strength}-pna-{pna}-{cfg.residuals_run_tag}.csv"
    )

    scores_path = data_dir / scores_filename
    residuals_path = data_dir / residuals_filename

    # Overwrite on each run to keep the script idempotent.
    with scores_path.open("w", newline="") as scores_f, residuals_path.open("w", newline="") as residuals_f:
        scores_writer = csv.writer(scores_f)
        residuals_writer = csv.writer(residuals_f)

        # Scores CSV header: designed to contain everything needed for later
        # SSMD/Z' bar plots.
        scores_writer.writerow([
            "layout",
            "error_type",
            "error",
            "E",
            "lost_rows",
            "neg_control_mean",
            "pos_control_mean",
            "neg_stdev",
            "pos_stdev",
            "Zfactor_expected",
            "SSMD_expected",
            "Zfactor_raw",
            "SSMD_raw",
            "Zfactor_norm",
            "SSMD_norm",
        ])

        # Residuals CSV header: compatible with utilities.plot_screening_plates,
        # plot_roc_curves, plot_pr_curves, pr_table_code, roc_table_code.
        residuals_writer.writerow([
            "layout",
            "error_type",
            "error",
            "E",
            "lost_rows",
            "neg_control_mean",
            "pos_control_mean",
            "neg_stdev",
            "pos_stdev",
            "comp_id",
            "true_residuals",
            "expected_result",
            "obtained_result",
            "activity",
            "batch",
            "plate_id",
        ])

        # E is currently unused in the screening notebook; fix to 0 for now.
        E = 0
        error_type = "bowl_nl"

        # For reproducibility, iterate layouts in a stable order.
        layouts = sorted(list(iter_layout_files(cfg)), key=lambda x: (x[0]["type"], x[1]))

        plate_counter = 0

        for plate_type, layout_file in layouts:
            layout_dir = cfg.root / plate_type["dir"]
            layout = np.load(layout_dir / layout_file)

            neg_control_id = np.max(layout)
            pos_control_id = neg_control_id - 1

            for plate_idx in range(cfg.plates_per_layout):
                # Batch index: utilities.* assume batches in [0, batches_for_curves-1]
                batch = plate_idx % cfg.batches_for_curves

                # Ideal plate and activity layout.
                ideal_plate, activity_layout = sc.fill_plate(
                    layout,
                    neg_control_id,
                    pos_control_id,
                    cfg.neg_control_mean,
                    cfg.pos_control_mean,
                    cfg.neg_stdev,
                    cfg.pos_stdev,
                    percent_non_active=1.0 - hit_rate,
                )

                # Expected metrics from the ideal plate (no disturbances).
                raw_neg_mean, raw_pos_mean, raw_neg_std, raw_pos_std = sc.control_stats(
                    ideal_plate, layout, neg_control_id, pos_control_id
                )
                ssmd_expected = sc.ssmd(raw_neg_mean, raw_pos_mean, raw_neg_std, raw_pos_std)
                z_expected = sc.zfactor(raw_neg_mean, raw_pos_mean, raw_neg_std, raw_pos_std)

                # Apply bowl-shaped error and (optionally) row loss.
                # For now, keep lost_rows at 0 to match the LaTeX figures that
                # filter on lost_rows < 1.
                lost_rows = 0
                plate_with_error = dt.add_bowlshaped_errors_nl(ideal_plate, error_strength)
                plate_with_error = dt.lose_rows(plate_with_error, 0, lost_rows)

                # Compute raw metrics on disturbed plate.
                raw_neg_mean2, raw_pos_mean2, raw_neg_std2, raw_pos_std2 = sc.control_stats(
                    plate_with_error, layout, neg_control_id, pos_control_id
                )
                ssmd_raw = sc.ssmd(raw_neg_mean2, raw_pos_mean2, raw_neg_std2, raw_pos_std2)
                z_raw = sc.zfactor(raw_neg_mean2, raw_pos_mean2, raw_neg_std2, raw_pos_std2)

                # Normalization: nearest-control scheme on the disturbed plate.
                neg_control_locations = util.get_controls_layout(layout.astype(np.float32))
                neg_control_locations = dt.lose_rows(neg_control_locations, 0, lost_rows)
                layout_after_loss = dt.lose_rows(layout, 0, lost_rows)

                norm_plate = nrm.normalize_plate_nearest_control(
                    plate_with_error, neg_control_locations, min_dist=0
                )

                norm_neg_mean, norm_pos_mean, norm_neg_std, norm_pos_std = sc.control_stats(
                    norm_plate, layout_after_loss, neg_control_id, pos_control_id
                )
                ssmd_norm = sc.ssmd(norm_neg_mean, norm_pos_mean, norm_neg_std, norm_pos_std)
                z_norm = sc.zfactor(norm_neg_mean, norm_pos_mean, norm_neg_std, norm_pos_std)

                # Write one scores row per plate.
                scores_writer.writerow([
                    plate_type["type"],
                    error_type,
                    error_strength,
                    E,
                    lost_rows,
                    raw_neg_mean,
                    raw_pos_mean,
                    raw_neg_std,
                    raw_pos_std,
                    z_expected,
                    ssmd_expected,
                    z_raw,
                    ssmd_raw,
                    z_norm,
                    ssmd_norm,
                ])

                # Residuals rows: one per well with activity / control status.
                num_rows, num_cols = layout.shape
                for r in range(num_rows):
                    for c in range(num_cols):
                        if layout[r, c] <= 0:
                            continue

                        comp_id = int(layout[r, c])
                        # Activity layout: 1 = active, 0 = inactive / control.
                        activity = int(activity_layout[r, c])

                        expected_val = ideal_plate[r, c]
                        obtained_val = plate_with_error[r, c]
                        true_residual = obtained_val - expected_val

                        residuals_writer.writerow([
                            plate_type["type"],        # layout
                            error_type,                # error_type
                            error_strength,            # error
                            E,                         # E
                            lost_rows,                 # lost_rows
                            raw_neg_mean,              # neg_control_mean
                            raw_pos_mean,              # pos_control_mean
                            raw_neg_std,               # neg_stdev
                            raw_pos_std,               # pos_stdev
                            comp_id,                   # comp_id
                            true_residual,             # true_residuals
                            expected_val,              # expected_result
                            obtained_val,              # obtained_result
                            activity,                  # activity
                            batch,                     # batch
                            plate_counter,             # plate_id
                        ])

                plate_counter += 1


def run_simulations(cfg: ScreeningConfig) -> None:
    """
    Run the full grid of screening simulations needed for:
    - expected vs obtained plots (bowl strengths 0.03, 0.06, 0.08 at 1% hit rate),
    - ROC/PR curves under strong bowl-shaped effects 0.2 for hit rates
      1, 5, 10, 20, 30, 40%.

    This deliberately over-simulates slightly (reusing conditions where needed)
    to keep the parameter grid explicit and easy to extend.
    """
    ensure_directories(cfg)

    # Expected vs obtained panels – use 1% hit rate.
    mild_hit_rate = 0.01
    for strength in cfg.bowl_strengths_expected_obtained:
        simulate_condition(cfg, error_strength=strength, hit_rate=mild_hit_rate)

    # Strong bowl-shaped effects for ROC/PR curves – full hit-rate grid.
    for hit_rate in cfg.hit_rates:
        simulate_condition(cfg, error_strength=cfg.bowl_strength_strong, hit_rate=hit_rate)


# ---------------------------------------------------------------------------
# Stage 2: Figures for supplement (expected vs obtained, ROC, PR)
# ---------------------------------------------------------------------------

def generate_expected_vs_obtained_figures(cfg: ScreeningConfig) -> None:
    """
    Generate expected-vs-obtained panels for mild/moderate/strong bowl strengths.

    These figures correspond to the screening_data_expected_obtained* entries
    and the top row of screening_data__paper in the artifact map.
    """
    ensure_directories(cfg)
    data_dir = cfg.root / cfg.data_dir
    fig_dir = cfg.root / cfg.figures_dir

    hit_rate = 0.01
    pna = 1.0 - hit_rate

    for strength in cfg.bowl_strengths_expected_obtained:
        residuals_filename = (
            f"screening-residuals-"
            f"{cfg.pos_controls}-{cfg.neg_controls}-"
            f"{strength}-pna-{pna}-{cfg.residuals_run_tag}.csv"
        )
        residuals_path = data_dir / residuals_filename

        # fig_name here is the "core" of the filename used by utilities.plot_screening_plates
        # so that it produces:
        #   screening-bowl-<strength>-10-10-0.99-stdev-3-4-*.png
        fig_name = f"{strength}-{cfg.pos_controls}-{cfg.neg_controls}-{pna}-stdev-3-4"

        util.plot_screening_plates(
            residuals_filename=str(residuals_path),
            fig_name=fig_name,
            fig_dir=str(fig_dir),
            max_value=450,
        )


def generate_roc_pr_figures(cfg: ScreeningConfig) -> None:
    """
    Generate ROC and PR curves for strong bowl-shaped effects (0.2) and
    hit rates 1, 5, 10, 20, 30, 40%.

    Filenames match those referenced in the tikz-figures screening_data_roc_strong
    and screening_data_pr_strong fragments, as well as the main-paper PR panels.
    """
    ensure_directories(cfg)
    data_dir = cfg.root / cfg.data_dir
    fig_dir = cfg.root / cfg.figures_dir

    strength = cfg.bowl_strength_strong

    # Mapping from hit rate to the "fig_name" suffix used in the original
    # supplemental plots: 1, 5, 10, 20, 30, 40.
    hit_to_suffix = {
        0.01: "1",
        0.05: "5",
        0.10: "10",
        0.20: "20",
        0.30: "30",
        0.40: "40",
    }

    # Batch selection list (mirroring the hand-tuned batch picks in the
    # original notebook; can be adjusted later if needed).
    # Here we simply use batch 0 for all; you can inject a custom mapping
    # per hit rate if you want exact visual replication.
    default_batch = 0

    for hit_rate in cfg.hit_rates:
        pna = 1.0 - hit_rate
        suffix = hit_to_suffix[hit_rate]

        residuals_filename = (
            f"screening-residuals-"
            f"{cfg.pos_controls}-{cfg.neg_controls}-"
            f"{strength}-pna-{pna}-{cfg.residuals_run_tag}.csv"
        )
        residuals_path = data_dir / residuals_filename

        roc_fig_name = f"ROC-{cfg.pos_controls}-{cfg.neg_controls}-{strength}-{suffix}.png"
        pr_fig_name = f"PR-{cfg.pos_controls}-{cfg.neg_controls}-{strength}-{suffix}.png"

        util.plot_roc_curves(
            residuals_filename=str(residuals_path),
            fig_name=roc_fig_name,
            fig_dir=str(fig_dir) + os.sep,
            batch=default_batch,
            batches=cfg.batches_for_curves,
        )

        util.plot_pr_curves(
            residuals_filename=str(residuals_path),
            fig_name=pr_fig_name,
            fig_dir=str(fig_dir) + os.sep,
            batch=default_batch,
            batches=cfg.batches_for_curves,
        )


# ---------------------------------------------------------------------------
# Stage 3: SSMD / Z' robustness metrics and figures (Group 6)
# ---------------------------------------------------------------------------

def run_metrics(cfg: ScreeningConfig) -> None:
    """
    Use the existing sc.test_quality_assessment_metrics helper to generate
    screening_metrics_data-10-10-<strength>-20250623-reviewing.csv files
    for the bowl-strength grid 0.00–0.08.

    To keep filenames stable, this temporarily patches sc.date.today() to
    return 2025-06-23, so that the 'today' stamp matches the artifact map.
    """
    ensure_directories(cfg)
    data_dir = cfg.root / cfg.data_dir

    # Patch sc.date.today() to a fixed date so that the filenames match
    # screening_metrics_data-10-10-<strength>-20250623-reviewing.csv
    original_date = sc.date

    class FixedDate(sc.date.__class__):
        @classmethod
        def today(cls):
            return original_date(2025, 6, 23)

    sc.date = FixedDate

    try:
        error_types = [
            {"type": "bowl_nl", "error_function": dt.add_bowlshaped_errors_nl},
        ]

        plate_types = [
            {
                "type": "Random",
                "dir": str(cfg.root / "layouts/screening_RANDM_layouts/") + os.sep,
                "regex": f"plate_layout_rand_{cfg.neg_controls}-{cfg.pos_controls}_.*.npy",
            },
            {
                "type": "PLAID",
                "dir": str(cfg.root / "layouts/screening_PLAID_layouts/") + os.sep,
                "regex": f"plate_layout_{cfg.neg_controls}-{cfg.pos_controls}_.*.npy",
            },
            {
                "type": "COMPD",
                "dir": str(cfg.root / "layouts/screening_COMPD_layouts/") + os.sep,
                "regex": f"plate_layout_{cfg.neg_controls}-{cfg.pos_controls}_.*.npy",
            },
        ]

        for strength in cfg.metrics_bowl_strengths:
            _ = sc.test_quality_assessment_metrics(
                plate_types=plate_types,
                error_types=error_types,
                error=strength,
                id_text=cfg.metrics_run_tag.split("-", 1)[1],  # "reviewing"
                neg_controls=cfg.neg_controls,
                pos_controls=cfg.pos_controls,
                neg_control_mean=cfg.neg_control_mean,
                pos_control_mean=cfg.pos_control_mean,
                neg_stdev=cfg.neg_stdev,
                pos_stdev=cfg.pos_stdev,
                data_directory=str(data_dir) + os.sep,
            )
    finally:
        # Restore original date object
        sc.date = original_date


def generate_metrics_figures(cfg: ScreeningConfig) -> None:
    """
    Generate SSMD and Z' MSE barplots over the bowl-strength grid, using
    utilities.plotting_residual_metrics. Filenames match the artifact map:

        figures/screening-SSMD-mse-screening_metrics_data-10-10-0.02-20250623-reviewing.csv.png
        figures/screening-Zfactor-mse-screening_metrics_data-10-10-0.02-20250623-reviewing.csv.png
    """
    ensure_directories(cfg)
    data_dir = cfg.root / cfg.data_dir
    fig_dir = cfg.root / cfg.figures_dir

    for strength in cfg.metrics_bowl_strengths:
        metrics_filename = (
            f"screening_metrics_data-"
            f"{cfg.neg_controls}-{cfg.pos_controls}-"
            f"{strength}-{cfg.metrics_run_tag}.csv"
        )
        metrics_path = data_dir / metrics_filename

        # Layout order Random, PLAID, COMPD as in the artifact map.
        order = ["Random", "PLAID", "COMPD"]
        box_pairs = [("Random", "PLAID"), ("Random", "COMPD"), ("PLAID", "COMPD")]

        # Z' factor MSE
        util.plotting_residual_metrics(
            screening_scores_data_filename=str(metrics_path),
            metric="Zfactor",
            fig_name=metrics_filename,
            y_min=None,
            y_max=None,
            palette=None,
            plots_directory=str(fig_dir) + os.sep,
            box_pairs=box_pairs,
            order=order,
        )

        # SSMD MSE
        util.plotting_residual_metrics(
            screening_scores_data_filename=str(metrics_path),
            metric="SSMD",
            fig_name=metrics_filename,
            y_min=None,
            y_max=None,
            palette=None,
            plots_directory=str(fig_dir) + os.sep,
            box_pairs=box_pairs,
            order=order,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COMPD/PLAID screening benchmark (screening side)."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "simulate", "figures", "metrics", "metrics-figures"],
        default="all",
        help="Which stage to run: simulations, figure generation, metrics, or all.",
    )
    parser.add_argument(
        "--data-dir",
        default="generated-data/screening",
        help="Directory for generated CSV data (relative to CWD).",
    )
    parser.add_argument(
        "--figures-dir",
        default="generated-plots/screening-supplement",
        help="Directory for generated figures (relative to CWD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ScreeningConfig(
        data_dir=Path(args.data_dir),
        figures_dir=Path(args.figures_dir),
    )

    if args.stage in ("all", "simulate"):
        run_simulations(cfg)

    if args.stage in ("all", "figures"):
        generate_expected_vs_obtained_figures(cfg)
        generate_roc_pr_figures(cfg)

    if args.stage in ("all", "metrics"):
        run_metrics(cfg)

    if args.stage in ("all", "metrics-figures"):
        generate_metrics_figures(cfg)


if __name__ == "__main__":
    main()