#!/usr/bin/env python3
"""Run the screening benchmark end-to-end.

Consolidates:
  - screening-experiments.ipynb (data generation)
  - screening-supplement.ipynb  (screening figures and ROC/PR plots)
  - plate-metrics-comparison.ipynb (SSMD/Z' robustness plots for bowl-nl)

Stages
------
  simulate : generate screening_scores_data-*.csv and screening-residuals-*.csv
  figures  : generate screening panels and ROC/PR plots
  metrics  : generate SSMD/Z' plots
  all      : simulate → figures → metrics  (default)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics

import libraries.disturbances as dt
import libraries.normalization as nrm
import libraries.screening as sc
import libraries.utilities as util
from benchmark_common import (
    SCREENING_LAYOUT_BOX_PAIRS,
    SCREENING_LAYOUT_ORDER,
    screening_metrics_plate_types,
    screening_plate_types,
    validate_layout_registry_consistency,
)
validate_layout_registry_consistency()

SCREENING_PANEL_CASES = [
    ("0.06-10-10-0.99-stdev-3-4", "screening-residuals-10-10-0.1-pna-0.99{today_tag}.csv", 450),
    ("0.08-10-10-0.99-stdev-3-4", "screening-residuals-10-10-0.2-pna-0.99{today_tag}.csv", 450),
    ("0.03-10-10-0.99-stdev-3-4", "screening-residuals-10-10-0.03-pna-0.99{today_tag}.csv", 450),
]

SCREENING_ROC_PR_CASES = [
    ("screening-residuals-10-10-0.2-pna-0.99{today_tag}.csv", "10-10-0.2-1.png", 6),
    ("screening-residuals-10-10-0.2-pna-0.95{today_tag}.csv", "10-10-0.2-5.png", 9),
    ("screening-residuals-10-10-0.2-pna-0.9{today_tag}.csv",  "10-10-0.2-10.png", 0),
    ("screening-residuals-10-10-0.2-pna-0.8{today_tag}.csv",  "10-10-0.2-20.png", 2),
    ("screening-residuals-10-10-0.2-pna-0.7{today_tag}.csv",  "10-10-0.2-30.png", 6),
    ("screening-residuals-10-10-0.2-pna-0.6{today_tag}.csv",  "10-10-0.2-40.png", None),
    ("screening-residuals-8-8-0.1-pna-0.99{today_tag}.csv",   "8-8-0.1-1.png", None),
    ("screening-residuals-8-8-0.1-pna-0.95{today_tag}.csv",   "8-8-0.1-5.png", 1),
    ("screening-residuals-8-8-0.1-pna-0.9{today_tag}.csv",    "8-8-0.1-10.png", 6),
    ("screening-residuals-8-8-0.1-pna-0.8{today_tag}.csv",    "8-8-0.1-20.png", 6),
    ("screening-residuals-8-8-0.1-pna-0.7{today_tag}.csv",    "8-8-0.1-30.png", 6),
    ("screening-residuals-8-8-0.1-pna-0.6{today_tag}.csv",    "8-8-0.1-40.png", 3),
]

CONTROL_LAYOUT_FIGURE_CASES = [
    ("layouts/screening_RANDM_layouts/plate_layout_rand_10-10_02.npy", "figures/plate_random-controls-rows-error.png"),
    ("layouts/screening_PLAID_layouts/plate_layout_10-10_01.npy", "figures/plate_plaid-controls-rows-error.png"),
    ("layouts/screening_COMPD_layouts/plate_layout_10-10_01.npy", "figures/plate_compd-controls-rows-error.png"),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlateType:
    type: str
    dir: str
    regex: str
    error_correction: callable


@dataclass
class ScreeningConfig:
    base_dir: Path = field(default_factory=lambda: Path("."))
    layouts_root: Path = field(default_factory=lambda: Path("layouts"))
    latex_tables_dir: Path = field(default_factory=lambda: Path("latex-tables"))

    screening_data_dir: Path = field(
        default_factory=lambda: Path("generated-data") / "screening"
    )
    screening_plots_dir: Path = field(
        default_factory=lambda: Path("generated-plots") / "screening-supplement"
    )
    metrics_data_dir: Path = field(
        default_factory=lambda: Path("generated-data") / "quality-assessment-metrics"
    )
    metrics_plots_dir: Path = field(
        default_factory=lambda: Path("generated-plots") / "quality-assessment-metrics"
    )

    neg_pos_controls_list: List[Tuple[int, int]] = field(
        default_factory=lambda: [(8, 8), (10, 10), (20, 10)]
    )
    error_strength_list: List[float] = field(
        default_factory=lambda: [0.03, 0.06, 0.1, 0.2]
    )
    hit_rate_list: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    )

    neg_control_mean: float = 100.0
    neg_stdev: float = 10.0
    pos_stdev: float = 10.0

    def plate_types(self, neg_controls: int, pos_controls: int) -> List[PlateType]:
        return [
            PlateType(
                type=plate_type["type"],
                dir=plate_type["dir"],
                regex=plate_type["regex"],
                error_correction=plate_type["error_correction"],
            )
            for plate_type in screening_plate_types(neg_controls, pos_controls)
        ]

    def error_types(self):
        return [{
            "type": "bowl-nl",
            "error_function": dt.add_bowlshaped_errors_nl,
            "error_correction": nrm.normalize_plate_nearest_control,
        }]

    id_text: str = "ROC-supplement"
    metrics_id_text: str = "reviewing"

    run_tag: str = "20250623-ROC-supplement"
    batches: int = 10
    lost_rows_range: Iterable[int] = field(default_factory=lambda: range(1, 4))

    @property
    def today_tag(self) -> str:
        return f"-{self.run_tag}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class _SkipPlate(Exception):
    """Raised internally to abort processing of one plate and move to the next."""


# ---------------------------------------------------------------------------
# Stage 1: Simulation
# ---------------------------------------------------------------------------

def simulate_condition(
    cfg: ScreeningConfig,
    neg_controls: int,
    pos_controls: int,
    error: float,
    percent_non_active: float,
) -> None:
    ensure_dir(cfg.screening_data_dir)

    scores_name = (
        f"screening_scores_data-{pos_controls}-{neg_controls}-{error}-pna-"
        f"{percent_non_active}{cfg.today_tag}.csv"
    )
    residuals_name = (
        f"screening-residuals-{pos_controls}-{neg_controls}-{error}-pna-"
        f"{percent_non_active}{cfg.today_tag}.csv"
    )

    scores_path = cfg.screening_data_dir / scores_name
    residuals_path = cfg.screening_data_dir / residuals_name

    # pos_control_mean is fixed at 40 (the original notebook swept range(40,41,120),
    # which is effectively just [40], so the loop is inlined here).
    pos_control_mean = 40

    with scores_path.open("w", newline="") as scores_f, \
         residuals_path.open("w", newline="") as residuals_f:

        scores_writer = csv.writer(scores_f)
        residuals_writer = csv.writer(residuals_f)

        scores_writer.writerow([
            "batch", "layout", "error_type", "error", "lost_rows",
            "neg_control_mean", "pos_control_mean", "neg_stdev", "pos_stdev",
            "Zfactor_expected", "SSMD_expected",
            "Zfactor_raw", "SSMD_raw",
            "Zfactor_norm", "SSMD_norm",
        ])
        residuals_writer.writerow([
            "batch", "layout", "error_type", "error", "lost_rows",
            "neg_control_mean", "pos_control_mean", "neg_stdev", "pos_stdev",
            "comp_id", "true_residuals", "expected_result",
            "obtained_result", "activity", "plate_id",
        ])

        plate_types = cfg.plate_types(neg_controls, pos_controls)

        for batch in range(cfg.batches):
            print(
                f"(neg,pos)=({neg_controls},{pos_controls}) "
                f"error={error} pna={percent_non_active} batch={batch}"
            )
            for plate_type in plate_types:
                layout_dir = plate_type.dir
                layouts = os.listdir(layout_dir)

                for layout_file in layouts:
                    match = re.search(plate_type.regex, layout_file)
                    if match is None:
                        continue

                    layout = np.load(os.path.join(layout_dir, layout_file))
                    neg_control_id = np.max(layout)
                    pos_control_id = neg_control_id - 1

                    for et in cfg.error_types():
                        for lost_rows in cfg.lost_rows_range:
                            limit = {"from": 1, "to": lost_rows}

                            try:
                                ideal_plate, activity_layout = sc.fill_plate(
                                    layout,
                                    neg_control_id,
                                    pos_control_id,
                                    cfg.neg_control_mean,
                                    pos_control_mean,
                                    neg_stdev=cfg.neg_stdev,
                                    pos_stdev=cfg.pos_stdev,
                                    percent_non_active=percent_non_active,
                                )

                                (
                                    exp_neg_mean, exp_pos_mean,
                                    exp_neg_std, exp_pos_std,
                                ) = sc.control_stats(
                                    ideal_plate, layout,
                                    neg_control_id, pos_control_id,
                                )
                                ssmd_expected = sc.ssmd(
                                    exp_neg_mean, exp_pos_mean,
                                    exp_neg_std, exp_pos_std,
                                )
                                zfactor_expected = sc.zfactor(
                                    exp_neg_mean, exp_pos_mean,
                                    exp_neg_std, exp_pos_std,
                                )

                                plate = et["error_function"](ideal_plate, error)
                                plate = dt.lose_rows(plate, limit["from"], limit["to"])

                                (
                                    raw_neg_mean, raw_pos_mean,
                                    raw_neg_std, raw_pos_std,
                                ) = sc.control_stats(
                                    plate, layout, neg_control_id, pos_control_id,
                                )
                                ssmd_raw = sc.ssmd(
                                    raw_neg_mean, raw_pos_mean,
                                    raw_neg_std, raw_pos_std,
                                )
                                zfactor_raw = sc.zfactor(
                                    raw_neg_mean, raw_pos_mean,
                                    raw_neg_std, raw_pos_std,
                                )

                                norm_plate = plate_type.error_correction(
                                    plate, layout, neg_control_id
                                )
                                remaining_layout = dt.lose_rows(
                                    layout, limit["from"], limit["to"]
                                )

                                (
                                    norm_neg_mean, norm_pos_mean,
                                    norm_neg_std, norm_pos_std,
                                ) = sc.control_stats(
                                    norm_plate, layout, neg_control_id, pos_control_id,
                                )
                                ssmd_norm = sc.ssmd(
                                    norm_neg_mean, norm_pos_mean,
                                    norm_neg_std, norm_pos_std,
                                )
                                zfactor_norm = sc.zfactor(
                                    norm_neg_mean, norm_pos_mean,
                                    norm_neg_std, norm_pos_std,
                                )

                            except Exception as exc:
                                raise _SkipPlate(
                                    f"Skipping {layout_file} batch={batch} "
                                    f"lost_rows={lost_rows}: {exc}"
                                ) from exc

                            scores_writer.writerow([
                                batch, plate_type.type, et["type"], error,
                                lost_rows - 1,
                                exp_neg_mean, exp_pos_mean,
                                exp_neg_std, exp_pos_std,
                                zfactor_expected, ssmd_expected,
                                zfactor_raw, ssmd_raw,
                                zfactor_norm, ssmd_norm,
                            ])

                            res_array = np.power(
                                np.reshape(
                                    np.abs(
                                        dt.lose_rows(ideal_plate, limit["from"], limit["to"])
                                        - plate
                                    ),
                                    (-1, 1),
                                ),
                                2,
                            )
                            comp_id_array = np.reshape(remaining_layout, (-1, 1))
                            ideal_plate_array = np.reshape(ideal_plate, (-1, 1))
                            norm_plate_array = np.reshape(plate, (-1, 1))
                            activity_array = np.reshape(activity_layout, (-1, 1))

                            comp_id_res_df = pd.DataFrame(
                                np.hstack([
                                    comp_id_array, res_array,
                                    ideal_plate_array, norm_plate_array,
                                    activity_array,
                                ]),
                                columns=[
                                    "comp_type", "res",
                                    "expected_result", "obtained_result",
                                    "activity",
                                ],
                            )
                            comp_id_res_df = comp_id_res_df[comp_id_res_df.comp_type > 0]

                            rrr = comp_id_res_df.to_numpy().T
                            _, res_size = rrr.shape

                            plate_residuals = np.vstack([
                                np.full(res_size, batch),
                                np.full(res_size, plate_type.type),
                                np.full(res_size, et["type"]),
                                np.full(res_size, error),
                                np.full(res_size, lost_rows - 1),
                                np.full(res_size, exp_neg_mean),
                                np.full(res_size, exp_pos_mean),
                                np.full(res_size, exp_neg_std),
                                np.full(res_size, exp_pos_std),
                                rrr,
                                np.full(res_size, match.group(2)),
                            ])

                            np.savetxt(
                                residuals_f,
                                plate_residuals.T,
                                delimiter=",",
                                fmt="%s",
                            )

    print("Done:", scores_path.name)
    print("Done:", residuals_path.name)


def run_simulations(cfg: ScreeningConfig) -> None:
    for neg_controls, pos_controls in cfg.neg_pos_controls_list:
        for error in cfg.error_strength_list:
            for percent_non_active in [1 - x for x in cfg.hit_rate_list]:
                simulate_condition(
                    cfg,
                    neg_controls=neg_controls,
                    pos_controls=pos_controls,
                    error=error,
                    percent_non_active=percent_non_active,
                )


# ---------------------------------------------------------------------------
# Stage 2: Screening figures
# ---------------------------------------------------------------------------

def generate_screening_panels(cfg: ScreeningConfig) -> None:
    """Expected vs obtained panels."""
    ensure_dir(cfg.screening_plots_dir)

    # Mapping between display label and simulation error level:
    #
    #   fig_name label | simulation error_strength | interpretation
    #   "0.03"         | 0.03                      | very mild bowl effect
    #   "0.06"         | 0.1                        | mild bowl effect
    #   "0.08"         | 0.2                        | moderate bowl effect
    #
    # The display labels follow the naming convention in the PLAID article
    # (Zhang 2008/2011 references), where mild ≈ 0.06 and strong ≈ 0.2
    # (see PLAID article §Methods). The simulation sweeps [0.03, 0.06, 0.1, 0.2];
    # the "0.06" figure uses simulation error=0.1 because visually that produces
    # the "mild" appearance described in the paper caption.
    # DO NOT change the CSV filenames without regenerating all screening data.
    for fig_name, residuals_file_template, max_value in SCREENING_PANEL_CASES:
        residuals_file = residuals_file_template.format(today_tag=cfg.today_tag)
        residuals_path = cfg.screening_data_dir / residuals_file
        util.plot_screening_plates(
            str(residuals_path),
            fig_name=fig_name,
            fig_dir=str(cfg.screening_plots_dir),
            max_value=max_value,
        )


def generate_roc_pr_curves(cfg: ScreeningConfig) -> None:
    """ROC / PR curves. Also prints roc_table_code / pr_table_code to stdout."""
    ensure_dir(cfg.screening_plots_dir)

    for residuals_file_template, fig_name, batch in SCREENING_ROC_PR_CASES:
        residuals_file = residuals_file_template.format(today_tag=cfg.today_tag)
        residuals_path = cfg.screening_data_dir / residuals_file
        kwargs = {} if batch is None else {"batch": batch}
        util.plot_roc_curves(str(residuals_path), "ROC-" + fig_name,
                             str(cfg.screening_plots_dir), **kwargs)
        util.plot_pr_curves(str(residuals_path), "PR-" + fig_name,
                            str(cfg.screening_plots_dir), **kwargs)
        util.roc_table_code(str(residuals_path))
        print("--------------")
        util.pr_table_code(str(residuals_path))


def generate_control_layout_figures(cfg: ScreeningConfig) -> None:
    """Control layout visualisations referenced by tikz-figures/a_figure_controls.tex."""
    ensure_dir(Path("figures"))

    neg_control_mean = 90
    pos_control_mean = 60
    neg_stdev = 2
    pos_stdev = 7

    np.random.seed(42)

    for layout_path, output_filename in CONTROL_LAYOUT_FIGURE_CASES:
        layout = np.load(layout_path)
        neg_control_id = np.max(layout)
        pos_control_id = neg_control_id - 1
        ideal_plate, _ = sc.fill_plate(
            layout, neg_control_id, pos_control_id,
            neg_control_mean, pos_control_mean, neg_stdev, pos_stdev,
        )
        disturbed_plate = dt.add_linear_errors_to_upper_rows_half(ideal_plate, 4)
        control_locations = util.get_controls_layout(layout)
        util.plot_plate(
            disturbed_plate,
            title="",
            mask=np.array(1 - control_locations, dtype=bool),
            filename=output_filename,
        )


def generate_screening_figures(cfg: ScreeningConfig) -> None:
    generate_control_layout_figures(cfg)
    generate_screening_panels(cfg)
    generate_roc_pr_curves(cfg)


# ---------------------------------------------------------------------------
# Stage 3: SSMD / Z' metrics
# ---------------------------------------------------------------------------

def run_metrics_simulation(cfg: ScreeningConfig) -> List[str]:
    ensure_dir(cfg.metrics_data_dir)

    output_file_list: List[str] = []
    id_text = cfg.metrics_id_text
    max_error = 26

    neg_control_mean = 100
    pos_control_mean = 5
    neg_stdev = 3
    pos_stdev = 4

    # NOTE: run_metrics_simulation uses its own plate_types dicts (not PlateType objects)
    # because sc.test_quality_assessment_metrics expects plain dicts with "type"/"dir"/"regex".
    # These are intentionally kept consistent with cfg.plate_types() — if you add a new layout,
    # update both places. This duplication will be removed when the layout registry lands.
    error_types = [{"type": "bowl-nl", "error_function": dt.add_bowlshaped_errors_nl}]
    data_directory = str(cfg.metrics_data_dir) + os.sep

    for neg_controls, pos_controls in cfg.neg_pos_controls_list:
        print(f"\nPlate {neg_controls}-{pos_controls}:")
        plate_types = screening_metrics_plate_types(neg_controls, pos_controls)

        for i in range(0, max_error):
            error = i / 100.0
            fname = sc.test_quality_assessment_metrics(
                plate_types, error_types, error, id_text,
                neg_controls, pos_controls,
                neg_control_mean, pos_control_mean,
                neg_stdev, pos_stdev,
                data_directory,
            )
            output_file_list.append(fname)
            print(f"  error={error:.2f} -> {fname}")

    return output_file_list


def generate_metrics_plots(cfg: ScreeningConfig, output_files: List[str]) -> None:
    ensure_dir(cfg.metrics_plots_dir)

    data_directory = str(cfg.metrics_data_dir) + os.sep
    plots_directory = str(cfg.metrics_plots_dir) + os.sep

    box_pairs = SCREENING_LAYOUT_BOX_PAIRS
    order = SCREENING_LAYOUT_ORDER

    for fname in output_files:
        for metric in ("Zfactor", "SSMD"):
            util.plotting_residual_metrics(
                data_directory + fname,
                metric=metric,
                fig_name=fname,
                y_max=None,
                palette=None,
                plots_directory=plots_directory,
                box_pairs=box_pairs,
                order=order,
            )

    manuscript_fname = "screening_metrics_data-10-10-0.06-20250623-reviewing.csv"
    for metric in ("Zfactor", "SSMD"):
        util.plotting_residual_metrics(
            data_directory + manuscript_fname,
            metric=metric,
            fig_name="manuscript",
            y_max=None, palette=None,
            plots_directory=plots_directory,
            box_pairs=box_pairs, order=order,
        )

    for neg_controls, pos_controls in cfg.neg_pos_controls_list:
        for i in range(0, 26):
            error = i / 100.0
            fname = (
                f"screening_metrics_data-{neg_controls}-{pos_controls}-"
                f"{error}-20250623-reviewing.csv"
            )
            util.plotting_residual_metrics(
                data_directory + fname,
                metric="Zfactor",
                fig_name=fname,
                y_max=0.04, palette=None,
                plots_directory=plots_directory,
                box_pairs=box_pairs, order=order,
            )
            util.plotting_residual_metrics(
                data_directory + fname,
                metric="SSMD",
                fig_name=fname,
                y_max=45, palette=None,
                plots_directory=plots_directory,
                box_pairs=box_pairs, order=order,
            )


def run_metrics(cfg: ScreeningConfig) -> None:
    files = run_metrics_simulation(cfg)
    generate_metrics_plots(cfg, files)


# ---------------------------------------------------------------------------
# Stage 4: Auto-generated LaTeX AUC table
# ---------------------------------------------------------------------------

def generate_auc_latex_table(cfg: ScreeningConfig) -> None:
    """
    Generates a LaTeX table with ROC-AUC and PR-AUC values across all hit rates,
    computed from all batches in the 10-10, error=0.2 screening residuals CSVs.

    The table replaces the hardcoded ROC-AUC / PR-AUC table in 0b_figures_tables.tex.
    Output: cfg.latex_tables_dir / screening_pr_10-10-0.2.tex
    """
    cfg.latex_tables_dir.mkdir(parents=True, exist_ok=True)

    hit_rates  = [1, 5, 10, 20, 30, 40]
    pna_values = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    layouts    = ["random", "plaid", "compd"]

    # Accumulate per-hit-rate results
    # summary[hit_rate][layout] = {"roc": (mean, std), "pr": (mean, std)}
    summary: dict = {}

    for hit_rate, pna in zip(hit_rates, pna_values):
        csv_name = (
            f"screening-residuals-10-10-0.2-pna-{pna}{cfg.today_tag}.csv"
        )
        csv_path = cfg.screening_data_dir / csv_name

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  WARNING: {csv_path} not found — skipping hit_rate={hit_rate}%")
            summary[hit_rate] = {
                lay: {"roc": (float("nan"), float("nan")),
                      "pr":  (float("nan"), float("nan"))}
                for lay in layouts
            }
            continue

        # Normalise column names (CSV header written by simulate_condition)
        df.columns = df.columns.str.strip()

        summary[hit_rate] = {}
        for layout in layouts:
            sub = df[df["layout"] == layout]
            if sub.empty:
                summary[hit_rate][layout] = {
                    "roc": (float("nan"), float("nan")),
                    "pr":  (float("nan"), float("nan")),
                }
                continue

            # Use all batches for statistical summary (the ROC/PR curve plots use a
            # single selected batch for visual clarity; this table uses all batches).
            roc_aucs, pr_aucs = [], []
            for _batch_id, grp in sub.groupby("batch"):
                y_true  = (grp["activity"].astype(float) > 0).astype(int)
                y_score = -grp["true_residuals"].astype(float)
                if y_true.nunique() < 2:
                    continue
                roc_aucs.append(skmetrics.roc_auc_score(y_true, y_score))
                pr_aucs.append(skmetrics.average_precision_score(y_true, y_score))

            summary[hit_rate][layout] = {
                "roc": (np.mean(roc_aucs), np.std(roc_aucs))
                       if roc_aucs else (float("nan"), float("nan")),
                "pr":  (np.mean(pr_aucs),  np.std(pr_aucs))
                       if pr_aucs  else (float("nan"), float("nan")),
            }

    # Build LaTeX table
    col_labels = [
        "Random ROC", "PLAID ROC", "COMPD ROC",
        "Random PR",  "PLAID PR",  "COMPD PR",
    ]
    lines = [
        r"\begin{tabular}{r" + "c" * 6 + r"}",
        r"\toprule",
        r"Hit rate & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for hit_rate in hit_rates:
        row = [f"{hit_rate}\\%"]
        for metric in ("roc", "pr"):
            for layout in layouts:
                mean, std = summary.get(hit_rate, {}).get(
                    layout, {}
                ).get(metric, (float("nan"), float("nan")))
                row.append("--" if np.isnan(mean) else f"{mean:.3f} $\\pm$ {std:.3f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    out_path = cfg.latex_tables_dir / "screening_pr_10-10-0.2.tex"
    out_path.write_text("\n".join(lines))
    print(f"  Written: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COMPD/PLAID screening benchmark pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["simulate", "figures", "metrics", "table", "all"],
        default="all",
        help=(
            "simulate: generate CSVs; "
            "figures: generate screening/ROC/PR plots; "
            "metrics: generate SSMD/Z' plots; "
            "table: generate LaTeX AUC table; "
            "all: simulate → figures → metrics → table"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScreeningConfig()

    if args.stage in ("simulate", "all"):
        run_simulations(cfg)
    if args.stage in ("figures", "all"):
        generate_screening_figures(cfg)
    if args.stage in ("metrics", "all"):
        run_metrics(cfg)
    if args.stage in ("table", "all"):
        generate_auc_latex_table(cfg)


if __name__ == "__main__":
    main()
