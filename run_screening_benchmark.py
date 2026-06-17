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
  tables   : generate LaTeX tables
  all      : from simulate to figures to metrics to tables (default)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional, Callable, Dict, Iterable, List, Tuple

import numpy as np
np.random.seed(42)
import pandas as pd
from scipy import stats as _scipy_stats
from sklearn import metrics as skmetrics

import libraries.disturbances as dt
import libraries.screening as sc
import libraries.utilities as util
from benchmark_common import (
    SCREENING_LAYOUT_BOX_PAIRS,
    SCREENING_LAYOUT_ORDER,
    SCREENING_LAYOUT_SPECS,
    screening_control_figure_cases,
    screening_metrics_plate_types,
    screening_plate_types,
    validate_layout_registry_consistency,
)
from benchmark_disturbances import (
    DISTURBANCES,
    dr_scenarios,
    screening_disturbances,
    _disturbance_function_for_screening_type,
)

validate_layout_registry_consistency()

def _build_screening_panel_cases() -> List[Tuple[str, str, int]]:
    cases = []
    for d in screening_disturbances():
        for lv in (d.screening_error_levels or ()):
            if lv.panel_neg_pos is None or lv.panel_fig_label is None:
                continue
            neg, pos = lv.panel_neg_pos
            fig_name = f"{lv.panel_fig_label}-{neg}-{pos}-0.99-stdev-3-4"
            csv_tmpl = f"screening-residuals-{neg}-{pos}-{lv.value}-pna-0.99{{today_tag}}.csv"
            cases.append((fig_name, csv_tmpl, 500, d.key))
    return cases

SCREENING_PANEL_CASES = _build_screening_panel_cases()

# Each entry: (residuals_file_template, fig_name_suffix).
# The batch index column has been removed — plot_roc_curves / plot_pr_curves
# now average over all batches automatically.
_SCREENING_HIT_RATE_PNA = [
    (1,  0.99),
    (5,  0.95),
    (10, 0.9),
    (20, 0.8),
    (30, 0.7),
    (40, 0.6),
]

def _build_screening_roc_pr_cases() -> List[Tuple[str, str]]:
    cases = []
    for d in screening_disturbances():
        for lv in (d.screening_error_levels or ()):
            if lv.panel_neg_pos is None:
                continue
            neg, pos = lv.panel_neg_pos
            for hit_rate, pna in _SCREENING_HIT_RATE_PNA:
                csv_tmpl = (
                    f"screening-residuals-{neg}-{pos}-{lv.value}"
                    f"-pna-{pna}{{today_tag}}.csv"
                )
                fig_suffix = f"{d.key}-{neg}-{pos}-{lv.value}-{hit_rate}.png"
                cases.append((csv_tmpl, fig_suffix, d.key))
    return cases

SCREENING_ROC_PR_CASES = _build_screening_roc_pr_cases()


# ── Caption templates (verbatim style from existing thin wrappers) ─────────
_SCREENING_CAPTION_PLACEHOLDER = "DISTURBANCE_DESC"

_SCREENING_CAPTIONS = {
    "panels": (
        r"Expected versus obtained screening results for "
        + _SCREENING_CAPTION_PLACEHOLDER
        + r". Panels show Random, PLAID, COMPD, and Expected (undisturbed) "
        r"layouts side by side, for mild, moderate, and strong disturbance levels."
    ),
    "roc": (
        r"ROC curves (mean $\pm$ 1\,s.d.\ across batches) for "
        + _SCREENING_CAPTION_PLACEHOLDER
        + r". Sub-panels correspond to hit rates of 1\%, 5\%, 10\%, "
        r"20\%, 30\%, and 40\%."
    ),
    "pr": (
        r"Precision-recall curves (mean $\pm$ 1\,s.d.\ across batches) for "
        + _SCREENING_CAPTION_PLACEHOLDER
        + r". Sub-panels correspond to hit rates of 1\%, 5\%, 10\%, "
        r"20\%, 30\%, and 40\%."
    ),
    "auc_overview": (
        r"AUC overview for "
        + _SCREENING_CAPTION_PLACEHOLDER
        + r". Rows: all control configurations $\times$ hit rates. "
        r"Best value per row is \textbf{bolded}; "
        r"COMPD significance vs.\ PLAID: "
        r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$."
    ),
}

_SCREENING_SUBSUBSECTION = {
    "panels":       r"Expected vs.\ obtained screening plates",
    "roc":          r"ROC curves",
    "pr":           r"Precision-recall curves",
    "auc_overview": r"AUC overview tables",
}

_METRIC_FULLNAME = {"roc": "ROC", "pr": "Precision-recall"}
_METRIC_CURVEDESC = {
    "roc": r"ROC curves (mean $\pm$ 1\,s.d.\ across batches)",
    "pr":  r"Precision-recall curves (mean $\pm$ 1\,s.d.\ across batches)",
}

# Error levels shown in SSMD/Z'-factor supplement figures.
# These match the manual wrappers screening_data_ssmd.tex and
# screening_data_z_factor.tex (errors 0.0 – 0.08 in steps of 0.01).
# The metrics simulation sweeps 0–0.25, but the supplement shows only this prefix.
_METRICS_DISPLAY_ERRORS: List[float] = [i / 100.0 for i in range(9)]  # 0.0 … 0.08
_METRICS_DISPLAY_CONTROLS: Tuple[int, int] = (10, 10)  # (neg, pos) shown in supplement

_CTRL_LABEL = {
    (8, 8):   "8 positive and 8 negative controls",
    (10, 10): "10 positive and 10 negative controls",
    (20, 10): "20 positive and 10 negative controls",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlateType:
    type: str          # lowercase key written into "layout" CSV column
    display_type: str  # canonical display label ("Random", "PLAID", "COMPD")
    dir: str
    regex: str
    error_correction: Callable


@dataclass
class ScreeningConfig:
    base_dir: Path = field(default_factory=lambda: Path("."))
    layouts_root: Path = field(default_factory=lambda: Path("layouts"))
    latex_tables_dir: Path = field(
        default_factory=lambda: Path("detailed-experimental-results-source") / "tables"
    )
    screening_plots_dir: Path = field(
        default_factory=lambda: Path("detailed-experimental-results-source") / "figures"
    )
    metrics_plots_dir: Path = field(
        default_factory=lambda: Path("detailed-experimental-results-source") / "figures"
    )

    screening_data_dir: Path = field(
        default_factory=lambda: Path("generated-data") / "screening"
    )
    metrics_data_dir: Path = field(
        default_factory=lambda: Path("generated-data") / "quality-assessment-metrics"
    )

    neg_pos_controls_list: List[Tuple[int, int]] = field(
        default_factory=lambda: [(8, 8), (10, 10), (20, 10)]
    )
    error_strength_list: List[float] = field(
        default_factory=lambda: sorted({
            lv.value
            for d in screening_disturbances()
            for lv in (d.screening_error_levels or ())
        })
    )
    hit_rate_list: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    )
    metrics_manuscript_controls: Tuple[int, int] = (10, 10)
    metrics_manuscript_error: float = 0.06
    neg_control_mean: float = 100.0
    neg_stdev: float = 10.0
    pos_stdev: float = 10.0

    def plate_types(self, neg_controls: int, pos_controls: int) -> List[PlateType]:
        """Build PlateType list for the screening simulation.

        Normalisation (error_correction) is per-layout, not per-disturbance.
        Each layout's correction is resolved from benchmark_common.SCREENING_LAYOUT_SPECS
        via LayoutSpec._resolved_error_correction(), defaults to normalize_plate_lowess_2d.
        To use a different correction for a specific layout, set error_correction
        on that layout's LayoutSpec entry in benchmark_common.py.
        """
        return [
            PlateType(
                type=plate_type["type"],
                display_type=plate_type["display_type"],
                dir=plate_type["dir"],
                regex=plate_type["regex"],
                error_correction=plate_type["error_correction"],
            )
            for plate_type in screening_plate_types(neg_controls, pos_controls)
        ]

    def error_types(self) -> List[Dict[str, Any]]:
        """Disturbance types for the screening simulation.

        Derived from the disturbance registry (benchmark_disturbances.py).
        Normalisation is per-layout, not per-disturbance: each PlateType carries
        its own error_correction callable.
        """
        return [
            {
                "type": d.screening_type,
                "disturbance_key": d.key,
                "error_function": _disturbance_function_for_screening_type(d.screening_type),
            }
            for d in screening_disturbances()
        ]

    id_text: str = "ROC-supplement"
    metrics_id_text: str = "reviewing"
    metrics_date_tag: str = "20250623"

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


def _unique_strength_levels(levels) -> list:
    """Return one representative ErrorLevel per logical strength label."""
    seen: set[str] = set()
    out = []
    for level in levels:
        if level.label in seen:
            continue
        seen.add(level.label)
        out.append(level)
    return out


def _screening_overview_table_stem(dist, metric: str, strength_label: str) -> str:
    """Filename stem for screening overview tables."""
    return f"screening-overview-{dist.key}-{metric}-{strength_label}"

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
        f"screening_scores_data-{neg_controls}-{pos_controls}-{error}-pna-"
        f"{percent_non_active}{cfg.today_tag}.csv"
    )
    residuals_name = (
        f"screening-residuals-{neg_controls}-{pos_controls}-{error}-pna-"
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
            "batch", "layout", "display_type",
            "error_type",  "disturbance_key", "error", "lost_rows",
            "neg_control_mean", "pos_control_mean", "neg_stdev", "pos_stdev",
            "Zfactor_expected", "SSMD_expected",
            "Zfactor_raw", "SSMD_raw",
            "Zfactor_norm", "SSMD_norm",
        ])
        
        # Note: "true_residuals" stores squared residuals vs the normalised plate,
        # and "obtained_result" is the normalised (error-corrected) signal,
        # matching the refactored plotting utilities (not the original notebook names).
        residuals_writer.writerow([
            "batch", "layout", "display_type",
            "error_type", "disturbance_key", "error", "lost_rows",
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
                layouts = sorted(os.listdir(layout_dir))

                for layout_file in layouts:
                    match = re.search(plate_type.regex, layout_file)
                    if match is None:
                        continue

                    try:
                        layout = np.load(os.path.join(layout_dir, layout_file))
                        neg_control_id = np.max(layout)
                        pos_control_id = neg_control_id - 1
                    except Exception as exc:
                        print(f"WARNING: could not load {layout_file!r}: {exc}")
                        continue

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

                                remaining_layout = dt.lose_rows(
                                    layout, limit["from"], limit["to"]
                                )
                                norm_plate = plate_type.error_correction(
                                    plate, remaining_layout, neg_control_id
                                )
                                (
                                    norm_neg_mean, norm_pos_mean,
                                    norm_neg_std, norm_pos_std,
                                ) = sc.control_stats(
                                    norm_plate, remaining_layout, neg_control_id, pos_control_id,
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
                                print(
                                    f"WARNING: skipping {layout_file!r} "
                                    f"batch={batch} lost_rows={lost_rows}: {exc}"
                                )
                                continue

                            scores_writer.writerow([
                                batch, plate_type.type, plate_type.display_type,
                                et["type"], et["disturbance_key"], error,
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
                            ideal_plate_array = np.reshape(
                                dt.lose_rows(ideal_plate, limit["from"], limit["to"]), (-1, 1)
                            )
                            norm_plate_array = np.reshape(norm_plate, (-1, 1))
                            activity_array = np.reshape(
                                dt.lose_rows(activity_layout, limit["from"], limit["to"]), (-1, 1)
                            )

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
                                np.full(res_size, plate_type.display_type),
                                np.full(res_size, et["type"]),
                                np.full(res_size, et["disturbance_key"]),
                                np.full(res_size, error),
                                np.full(res_size, lost_rows - 1),
                                np.full(res_size, exp_neg_mean),
                                np.full(res_size, exp_pos_mean),
                                np.full(res_size, exp_neg_std),
                                np.full(res_size, exp_pos_std),
                                rrr,
                                np.full(res_size, Path(layout_file).stem.rsplit("_", 1)[-1]),
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
    #   fig_name label | simulation error_strength  | interpretation
    #   "0.03"         | 0.06                       | mild bowl effect
    #   "0.06"         | 0.1                        | moderate bowl effect
    #   "0.08"         | 0.2                        | strong bowl effect
    #
    # The display labels follow the naming convention in the PLAID article
    # (Zhang 2008/2011 references), where mild ≈ 0.06 and strong ≈ 0.2
    # (see PLAID article §Methods). The simulation sweeps [0.06, 0.1, 0.2];
    # the "0.06" figure uses simulation error=0.1 because visually that produces
    # the "mild" appearance described in the paper caption.
    # DO NOT change the CSV filenames without regenerating all screening data.
    for fig_name, residuals_file_template, max_value, dist_key in SCREENING_PANEL_CASES:
        residuals_file = residuals_file_template.format(today_tag=cfg.today_tag)
        residuals_path = cfg.screening_data_dir / residuals_file
        util.plot_screening_plates(
            str(residuals_path),
            fig_name=fig_name,
            fig_dir=str(cfg.screening_plots_dir),
            max_value=max_value,
            dist_key=dist_key,
        )


def generate_roc_pr_curves(cfg: ScreeningConfig) -> None:
    """ROC / PR curves averaged over all batches (mean +/- 1-std band)."""
    ensure_dir(cfg.screening_plots_dir)

    for residuals_file_template, fig_name, dist_key in SCREENING_ROC_PR_CASES:
        residuals_file = residuals_file_template.format(today_tag=cfg.today_tag)
        residuals_path = cfg.screening_data_dir / residuals_file
        util.plot_roc_curves(
            str(residuals_path),
            "ROC-" + fig_name,
            str(cfg.screening_plots_dir),
            dist_key=dist_key
        )
        util.plot_pr_curves(
            str(residuals_path),
            "PR-" + fig_name,
            str(cfg.screening_plots_dir),
            dist_key=dist_key
        )


def generate_control_layout_figures(cfg: ScreeningConfig) -> None:
    """Control layout visualisations referenced by tikz-figures/a_figure_controls.tex."""
    ensure_dir(Path("detailed-experimental-results-source") / "figures")

    neg_control_mean = 90
    pos_control_mean = 60
    neg_stdev = 2
    pos_stdev = 7

    np.random.seed(42)
    # Generate one shared base plate using an arbitrary reference layout shape
    ref_layout = np.load(next(iter(screening_control_figure_cases()))[0])
    neg_control_id = np.max(ref_layout)
    pos_control_id = neg_control_id - 1
    shared_ideal, _ = sc.fill_plate(
        ref_layout, neg_control_id, pos_control_id,
        neg_control_mean, pos_control_mean, neg_stdev, pos_stdev,
    )
    shared_disturbed = dt.add_linear_errors_to_upper_rows_half(shared_ideal, 4)
    
    vmin = float(shared_disturbed.min())
    vmax = float(shared_disturbed.max())
    
    util.plot_plate(
        shared_disturbed,
        title="",
        mask=None,
        filename="detailed-experimental-results-source/figures/plate_rows-error-base.png",
        vmin=vmin, vmax=vmax
    )

    for layout_path, output_filename in screening_control_figure_cases():
        layout = np.load(layout_path)
        neg_control_id = np.max(layout)
        control_locations = util.get_controls_layout(layout)
        util.plot_plate(
            shared_disturbed,
            title="",
            mask=np.array(1 - control_locations, dtype=bool),
            filename=output_filename,
            vmin=vmin, vmax=vmax
        )


def generate_screening_figures(cfg: ScreeningConfig) -> None:
    generate_control_layout_figures(cfg)
    generate_screening_panels(cfg)
    generate_roc_pr_curves(cfg)


# ---------------------------------------------------------------------------
# Stage 3: SSMD / Z' metrics
# ---------------------------------------------------------------------------

def run_metrics_simulation(cfg: ScreeningConfig) -> List[str]:
    np.random.seed(42)
    ensure_dir(cfg.metrics_data_dir)

    output_file_list: List[str] = []
    id_text = cfg.metrics_id_text
    max_error = 26

    neg_control_mean = 100
    pos_control_mean = 5
    neg_stdev = 3
    pos_stdev = 4

    data_directory = str(cfg.metrics_data_dir) + os.sep

    for neg_controls, pos_controls in cfg.neg_pos_controls_list:
        print(f"\nPlate {neg_controls}-{pos_controls}:")
        plate_types = screening_metrics_plate_types(neg_controls, pos_controls)

        for d in screening_disturbances():
            # One error_type at a time so the CSV filename carries the dist key
            error_types_single = [
                {
                    "type": d.screening_type,
                    "disturbance_key": d.key,
                    "error_function": _disturbance_function_for_screening_type(
                        d.screening_type
                    ),
                }
            ]
            for i in range(0, max_error):
                error = i / 100.0
                fname = sc.test_quality_assessment_metrics(
                    plate_types, error_types_single, error,
                    f"{d.key}-{id_text}",           # dist key in run_tag / filename
                    neg_controls, pos_controls,
                    neg_control_mean, pos_control_mean,
                    neg_stdev, pos_stdev,
                    data_directory,
                    run_tag=f"{cfg.metrics_date_tag}-{d.key}-{id_text}",
                )
                output_file_list.append(fname)
                print(f"  [{d.key}] error={error:.2f} -> {fname}")

    return output_file_list


def generate_metrics_plots(cfg: ScreeningConfig, output_files: List[str]) -> None:
    ensure_dir(cfg.metrics_plots_dir)
    
    data_directory = str(cfg.metrics_data_dir) + os.sep
    plots_directory = str(cfg.metrics_plots_dir) + os.sep
    
    box_pairs = SCREENING_LAYOUT_BOX_PAIRS
    order = SCREENING_LAYOUT_ORDER

    neg_m, pos_m = cfg.metrics_manuscript_controls
    err_m = cfg.metrics_manuscript_error
    
    pattern = f"screening_metrics_data-{neg_m}-{pos_m}-{err_m}"
    
    try:
        manuscript_fname = next(
            f for f in output_files
            if pattern in f
        )
    except StopIteration:
        raise RuntimeError(
            f"Could not find metrics manuscript file matching {pattern!r} "
            f"in {len(output_files)} output files"
        )
    
    # ── Manuscript figures: bowl_nl_neg_unaffected only ──────────────────
    # screening_data__paper.tex hardcodes these filenames; do not regenerate
    # them for other disturbances.
    MANUSCRIPT_DISTURBANCE = "bowl_nl_neg_unaffected"
    manuscript_dist = next(
        (d for d in screening_disturbances() if d.key == MANUSCRIPT_DISTURBANCE), None
    )
    if manuscript_dist is not None:
        neg_m, pos_m = cfg.metrics_manuscript_controls
        err_m = cfg.metrics_manuscript_error
        pattern = f"screening_metrics_data-{neg_m}-{pos_m}-{err_m}"
        # Only files that belong to the bowl_nl_neg_unaffected simulation pass
        # through here; the CSV is per-(neg,pos,error), not per-disturbance,
        # so we pick the file and rely on the fact that bowl is first in the
        # loop — see Fix B below which segregates them.
        manuscript_files = [f for f in output_files if pattern in f
                            and MANUSCRIPT_DISTURBANCE in f]  # after Fix B
        if manuscript_files:
            for metric in ("Zfactor", "SSMD"):
                util.plotting_residual_metrics(
                    data_directory + manuscript_files[0],
                    metric=metric,
                    fig_name="manuscript",
                    y_max=None, palette=None,
                    plots_directory=plots_directory,
                    box_pairs=box_pairs, order=order,
                )
    # ── Per-disturbance figures ───────────────────────────────────────────
    for fname in output_files:
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

def _collect_per_batch_aucs_for(
    cfg: "ScreeningConfig",
    neg_controls: int,
    pos_controls: int,
    error: float,
    dist_key: Optional[str],
) -> dict:
    """Return {hit_rate: {layout: {"roc": list, "pr": list}}} for one scenario.

    If dist_key is given and the CSV contains a 'disturbance_key' column,
    only rows matching dist_key are used.
    """
    hit_rates  = [1, 5, 10, 20, 30, 40]
    pna_values = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    layouts    = SCREENING_LAYOUT_ORDER

    result: dict = {}
    for hit_rate, pna in zip(hit_rates, pna_values):
        csv_name = (
            f"screening-residuals-{neg_controls}-{pos_controls}-{error}"
            f"-pna-{pna}{cfg.today_tag}.csv"
        )
        csv_path = cfg.screening_data_dir / csv_name
        result[hit_rate] = {lay: {"roc": [], "pr": []} for lay in layouts}
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  WARNING: {csv_path} not found — skipping hit_rate={hit_rate}%")
            continue
        df.columns = df.columns.str.strip()

        # Filter by disturbance_key if the column exists and a key was provided.
        if dist_key is not None and "disturbance_key" in df.columns:
            df = df[df["disturbance_key"] == dist_key]

        for layout in layouts:
            sub = df[df["display_type"] == layout]
            for _batch_id, grp in sub.groupby("batch"):
                y_true  = (grp["activity"].astype(float) > 0).astype(int)
                y_score = -grp["obtained_result"].astype(float)
                if y_true.nunique() < 2:
                    continue
                result[hit_rate][layout]["roc"].append(
                    skmetrics.roc_auc_score(y_true, y_score)
                )
                result[hit_rate][layout]["pr"].append(
                    skmetrics.average_precision_score(y_true, y_score)
                )
    return result


def _collect_per_batch_aucs(cfg: "ScreeningConfig") -> dict:
    """Backwards-compatible wrapper for the primary 10-10-0.2 scenario."""
    return _collect_per_batch_aucs_for(cfg, neg_controls=10, pos_controls=10, error=0.2)


def _auc_summary(per_batch: dict, metric: str) -> dict:
    """
    From per_batch {hit_rate: {layout: {metric: list}}},
    return {hit_rate: {layout: (mean, std)}}.
    """
    summary = {}
    for hr, layout_dict in per_batch.items():
        summary[hr] = {}
        for lay, metrics in layout_dict.items():
            vals = metrics[metric]
            summary[hr][lay] = (
                (float("nan"), float("nan"))
                if not vals
                else (float(np.mean(vals)), float(np.std(vals)))
            )
    return summary


def generate_auc_latex_tables(cfg: "ScreeningConfig") -> None:
    """Generate ROC-AUC and PR-AUC LaTeX table fragments for every
    (neg_controls, pos_controls, error) scenario in cfg.

    For each scenario four files are written to cfg.latex_tables_dir:
      screening-roc-auc-{neg}-{pos}-{error}.tex
      screening-roc-pvalues-{neg}-{pos}-{error}.tex
      screening-pr-auc-{neg}-{pos}-{error}.tex
      screening-pr-pvalues-{neg}-{pos}-{error}.tex

    For the 10-10-0.2 scenario the combined main-paper table is also written:
      screening_pr_10-10-0.2.tex  (imported by 0b_figures_tables.tex)
    """
    cfg.latex_tables_dir.mkdir(parents=True, exist_ok=True)
    d  = cfg.latex_tables_dir
    hr = [1, 5, 10, 20, 30, 40]
    ly = SCREENING_LAYOUT_ORDER
    
    for dist in screening_disturbances():
        levels = _unique_strength_levels(dist.screening_error_levels or ())
        for level in levels:
            error = level.value
            for neg, pos in cfg.neg_pos_controls_list:
                tag       = f"{dist.key}-{neg}-{pos}-{error}"
                per_batch = _collect_per_batch_aucs_for(cfg, neg, pos, error, dist_key=dist.key)
                roc_summ  = _auc_summary(per_batch, "roc")
                pr_summ   = _auc_summary(per_batch, "pr")

                util.write_latex_auc_table(
                    roc_summ, hr, ly, d / f"screening-roc-auc-{tag}.tex"
                )
                util.write_latex_auc_table(
                    pr_summ,  hr, ly, d / f"screening-pr-auc-{tag}.tex"
                )
                util.write_latex_auc_pvalue_table(
                    per_batch, "roc", hr, d / f"screening-roc-pvalues-{tag}.tex"
                )
                util.write_latex_auc_pvalue_table(
                    per_batch, "pr",  hr, d / f"screening-pr-pvalues-{tag}.tex"
                )


def generate_auc_overview_tables(cfg: "ScreeningConfig") -> None:
    """Generate overview ROC-AUC and PR-AUC tables split by disturbance and strength."""
    from scipy import stats as _st

    cfg.latex_tables_dir.mkdir(parents=True, exist_ok=True)
    layouts   = SCREENING_LAYOUT_ORDER
    hit_rates = [1, 5, 10, 20, 30, 40]
    
    def _sig_flag(a_vals: list, b_vals: list) -> str:
        """Return LaTeX superscript for COMPD-vs-PLAID significance."""
        if len(a_vals) < 2 or len(b_vals) < 2:
            return ""
        _, p = _st.ttest_ind(a_vals, b_vals, equal_var=False)
        if p < 0.001:
            return r"$^{***}$"
        if p < 0.01:
            return r"$^{**}$"
        if p < 0.05:
            return r"$^{*}$"
        return ""

    for dist in screening_disturbances():
        levels = _unique_strength_levels(dist.screening_error_levels or ())

        for level in levels:
            error = level.value
            strength_label = level.label

            for metric in ("roc", "pr"):
                col_spec = "ll" + "c" * len(layouts)
                lines = [
                    rf"\begin{{tabular}}{{{col_spec}}}",
                    r"\toprule",
                    r"Config & Hit rate & " + " & ".join(layouts) + r" \\",
                    r"\midrule",
                ]

                first_group = True
                for neg, pos in cfg.neg_pos_controls_list:
                    per_batch = _collect_per_batch_aucs_for(cfg, neg, pos, error, dist_key=dist.key)
                    summ      = _auc_summary(per_batch, metric)
                    group_label = f"{neg}--{pos}"

                    if not first_group:
                        lines.append(r"\midrule")
                    first_group = False

                    for i, hr in enumerate(hit_rates):
                        cfg_cell = group_label if i == 0 else ""

                        means = [summ[hr][lay][0] for lay in layouts]
                        valid_means = [m for m in means if not np.isnan(m)]
                        best = max(valid_means) if valid_means else float("nan")

                        row = [cfg_cell, f"{hr}\\%"]
                        for lay in layouts:
                            mean, std = summ[hr][lay]
                            if np.isnan(mean):
                                cell = "--"
                            else:
                                cell = util.fmt_mean_std(mean, std, bold=(mean == best))
                                if lay == "COMPD":
                                    plaid_vals = per_batch[hr].get("PLAID", {}).get(metric, [])
                                    compd_vals = per_batch[hr].get("COMPD", {}).get(metric, [])
                                    cell += _sig_flag(plaid_vals, compd_vals)
                            row.append(cell)

                        lines.append(" & ".join(row) + r" \\")

                lines += [r"\bottomrule", r"\end{tabular}"]

                out_path = cfg.latex_tables_dir / (
                    _screening_overview_table_stem(dist, metric, strength_label) + ".tex"
                )
                out_path.write_text("\n".join(lines))
                print(f"  Written: {out_path}")


def generate_metrics_latex_tables(
    cfg: "ScreeningConfig",
    output_files: "list[str] | None" = None,
) -> None:
    """Generate SSMD and Z-factor MSE LaTeX table fragments for every
    (neg_controls, pos_controls) pair across representative error levels.

    Four files per (neg, pos) × metric are written to cfg.latex_tables_dir:
      metrics-zfactor-mean-std-{neg}-{pos}.tex
      metrics-zfactor-pvalues-{neg}-{pos}.tex
      metrics-ssmd-mean-std-{neg}-{pos}.tex
      metrics-ssmd-pvalues-{neg}-{pos}.tex

    Parameters
    ----------
    output_files :
        List of metrics CSV basenames (no directory).  When ``None`` the
        directory ``cfg.metrics_data_dir`` is globbed for ``*.csv`` files.
    """
    cfg.latex_tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir = cfg.metrics_data_dir
    d        = cfg.latex_tables_dir
    ly       = SCREENING_LAYOUT_ORDER

    if output_files is None:
        output_files = [p.name for p in sorted(data_dir.glob("*.csv"))]
        if not output_files:
            print("  WARNING: no metrics CSV files found — run --stage metrics first")
            return

    # Representative error levels shown in tables (subset of the full sweep)
    representative_errors = cfg.error_strength_list  # e.g. [0.06, 0.10, 0.20]
    
    tables_dir = cfg.latex_tables_dir
    
    for neg, pos in cfg.neg_pos_controls_list:
        tag = f"{neg}-{pos}"
        
        for d in screening_disturbances():
            for metric in ("Zfactor", "SSMD"):
                metric_files: "list[str]" = []
                found_levels: "list[float]" = []
                for elevel in representative_errors:
                    candidates = [
                        f for f in output_files
                        if f"-{neg}-{pos}-" in f
                        and str(elevel) in f
                        and d.key in f
                    ]
                    if candidates:
                        metric_files.append(str(data_dir / candidates[0]))
                        found_levels.append(elevel)
                    else:
                        print(f"  WARNING: no metrics file for ({neg},{pos}) "
                              f"dist={d.key} error={elevel}")

                if not metric_files:
                    print(f"  Skipping metrics tables for ({neg},{pos}) — no files found")
                    continue

                for kind in ("mean-std", "pvalues"):
                    util.write_latex_metrics_table(
                        data_files=metric_files,
                        error_levels=found_levels,
                        metric=metric,
                        layouts=ly,
                        path=tables_dir / f"metrics-{metric.lower()}-{kind}-{tag}-{d.key}.tex",
                        kind=kind,
                    )

def generate_screening_section_tex(cfg: "ScreeningConfig") -> None:
    """Write tikz-figures/screening_section_auto.tex.

    AUTO-GENERATED — do not edit by hand.
    Regenerate with:  python run_screening_benchmark.py --stage tables

    Section structure per published disturbance (currently: bowl-shaped):

        \\subsection{\\emph{bowl-shaped} plate effects (screening)}
          \\subsubsection{Expected vs. obtained screening plates}
            — one figure* per strength level (3 subfigs: Random, PLAID, COMPD)
          \\subsubsection{ROC curves}
            — one figure* per (strength × control config), 6 subfigs 3×2 grid
          \\subsubsection{Precision-recall curves}
            — same layout as ROC curves
          \\subsubsection{AUC overview tables}
            — one table* per (metric × strength): ROC-mild, PR-mild, … ROC-strong, PR-strong
    """
    from pathlib import Path

    tikz_dir = Path("detailed-experimental-results-source") / "tikz-figures"
    tikz_dir.mkdir(parents=True, exist_ok=True)
    out_path = tikz_dir / "screening_section_auto.tex"

    fig_root = Path("detailed-experimental-results-source") / "figures"
    tbl_root = Path("detailed-experimental-results-source") / "tables"

    def _fig_exists(name: str) -> bool:
        return (fig_root / name).exists()

    def _tbl_exists(stem: str) -> bool:
        return (tbl_root / f"{stem}.tex").exists()

    lines: List[str] = [
        "% AUTO-GENERATED by run_screening_benchmark.py -- DO NOT EDIT.",
        "% Regenerate with:  python run_screening_benchmark.py --stage tables",
        "%",
    ]

    for d in screening_disturbances():
        emph    = d.emph_name
        levels  = d.screening_error_levels or ()
        strength_levels = _unique_strength_levels(levels)

        lines += [
            "",
            rf"\subsection{{\emph{{{emph}}} plate effects (screening)}}",
            "",
        ]

        # ── 2.x.1  Expected vs. obtained screening plates ────────────────
        # One figure* per strength level, 3 subfigs (Random, PLAID, COMPD).
        # Layout: three 0.31\textwidth panels side by side with \hfill.
        lines += [r"\subsubsection{Expected vs.\ obtained screening plates}", ""]

        for level in levels:
            if level.panel_neg_pos is None or level.panel_fig_label is None:
                continue
            neg, pos = level.panel_neg_pos
            label    = f"fig:screening-data-{level.label}"
            layout_enum = ", ".join(
                spec.display_type for spec in SCREENING_LAYOUT_SPECS
            )
            cap = (
                rf"Expected versus obtained screening results for "
                rf"\emph{{{emph}}} plate effects ({level.label} strength, "
                rf"{neg}\,--\,{pos} controls). "
                rf"Panels show {layout_enum} layouts "
                rf"after error correction and normalisation, "
                rf"with 1\% hit rate and 1~replicate per compound."
            )

            lines.append(r"\begin{figure*}[!htb]")
            lines.append(r"  \centering")
            layout_display_names = [spec.display_type for spec in SCREENING_LAYOUT_SPECS]
            
            for i, spec in enumerate(SCREENING_LAYOUT_SPECS):
                png = f"screening-{d.key}-{level.panel_fig_label}-{neg}-{pos}-0.99-stdev-3-4-{spec.key}.png"
                n_cols = len(SCREENING_LAYOUT_SPECS)
                if _fig_exists(png):
                    inc = rf"\includegraphics[width=\textwidth]{{figures/{png}}}"
                else:
                    inc = f"% MISSING: figures/{png}"
                lines += [
                    rf"  \begin{{subfigure}}[b]{{{util.subfigure_col_width(n_cols)}\textwidth}}",
                    r"    \centering",
                    rf"    \caption{{{spec.display_type}}}",
                    f"    {inc}",
                    r"  \end{subfigure}",
                ]
                if i < 2:
                    lines.append(r"  \hfill")
            lines += [
                rf"  \caption{{{cap}}}",
                rf"  \label{{{label}}}",
                r"\end{figure*}",
                "",
            ]

        lines.append(r"\clearpage")
        lines.append("")

        # ── 2.x.2  ROC curves ────────────────────────────────────────────
        # One figure* per (strength × control config).
        # Layout: 6 subfigs in 3 rows × 2 cols, each 0.49\textwidth with \hfill.
        lines += [r"\subsubsection{ROC curves}", ""]

        for level in strength_levels:
            for (neg, pos) in cfg.neg_pos_controls_list:
                ctrl_desc = _CTRL_LABEL.get((neg, pos), f"{neg} -- {pos} controls")
                label = f"fig:screening-roc-{d.key}-{neg}-{pos}-{level.label}"
                cap = (
                    rf"Comparison of {_METRIC_CURVEDESC['roc']} for "
                    rf"\emph{{{emph}}} plate effects ({level.label} strength) "
                    rf"with different hit rates, "
                    rf"using layouts with {ctrl_desc}."
                )

                lines.append(r"\begin{figure*}[!htb]")

                hit_rate_pna = list(_SCREENING_HIT_RATE_PNA)  # 6 items
                for row in range(3):
                    left_hr,  _ = hit_rate_pna[row * 2]
                    right_hr, _ = hit_rate_pna[row * 2 + 1]
                    for side, hr in [("left", left_hr), ("right", right_hr)]:
                        png = f"ROC-{d.key}-{neg}-{pos}-{level.value}-{hr}.png"
                        lines += [
                            rf"  \begin{{subfigure}}[b]{{{util.subfigure_col_width(2)}\textwidth}}",
                            r"    \centering",
                            rf"    \includegraphics[width=\textwidth]{{figures/{png}}}" if _fig_exists(png) else f"    % MISSING: figures/{png}",
                            rf"    \caption{{{hr}\% hits}}",
                            r"  \end{subfigure}",
                        ]
                        if side == "left":
                            lines.append(r"  \hfill")
                    lines.append("")  # blank line between rows

                lines += [
                    rf"  \caption{{{cap}}}",
                    rf"  \label{{{label}}}",
                    r"\end{figure*}",
                    "",
                ]

        lines.append(r"\clearpage")
        lines.append("")

        # ── 2.x.3  Precision-recall curves ──────────────────────────────
        # Identical layout to ROC.
        lines += [r"\subsubsection{Precision-recall curves}", ""]

        for level in strength_levels:
            for (neg, pos) in cfg.neg_pos_controls_list:
                ctrl_desc = _CTRL_LABEL.get((neg, pos), f"{neg} -- {pos} controls")
                label = f"fig:screening-pr-{d.key}-{neg}-{pos}-{level.label}"
                cap = (
                    rf"Comparison of {_METRIC_CURVEDESC['pr']} for "
                    rf"\emph{{{emph}}} plate effects ({level.label} strength) "
                    rf"with different hit rates, "
                    rf"using layouts with {ctrl_desc}."
                )

                lines.append(r"\begin{figure*}[!htb]")

                hit_rate_pna = list(_SCREENING_HIT_RATE_PNA)
                for row in range(3):
                    left_hr,  _ = hit_rate_pna[row * 2]
                    right_hr, _ = hit_rate_pna[row * 2 + 1]
                    for side, hr in [("left", left_hr), ("right", right_hr)]:
                        png = f"PR-{d.key}-{neg}-{pos}-{level.value}-{hr}.png"
                        lines += [
                            rf"  \begin{{subfigure}}[b]{{{util.subfigure_col_width(2)}\textwidth}}",
                            r"    \centering",
                            rf"    \includegraphics[width=\textwidth]{{figures/{png}}}" if _fig_exists(png) else f"    % MISSING: figures/{png}",
                            rf"    \caption{{{hr}\% hits}}",
                            r"  \end{subfigure}",
                        ]
                        if side == "left":
                            lines.append(r"  \hfill")
                    lines.append("")

                lines += [
                    rf"  \caption{{{cap}}}",
                    rf"  \label{{{label}}}",
                    r"\end{figure*}",
                    "",
                ]

        lines.append(r"\clearpage")
        lines.append("")

        # ── 2.x.4  AUC overview tables ───────────────────────────────────
        # Order: ROC-mild, PR-mild, ROC-moderate, PR-moderate, ROC-strong, PR-strong.
        # Each is a table* (full-width) with \small, booktabs, proper caption and \label.
        lines += [r"\subsubsection{AUC overview tables}", ""]

        for level in strength_levels:
            for metric in ("roc", "pr"):
                stem  = _screening_overview_table_stem(d, metric, level.label)
                label = f"tab:screening-{d.key}-{metric}-{level.label}"
                cap = (
                    rf"{_METRIC_FULLNAME[metric]}~AUC overview for "
                    rf"\emph{{{emph}}} plate effects ({level.label} strength). "
                    rf"Rows: all control configurations $\times$ hit rates. "
                    rf"Best value per row is \textbf{{bolded}}; "
                    rf"COMPD significance vs.\ PLAID: "
                    rf"$^{{*}}p{{<}}0.05$, $^{{**}}p{{<}}0.01$, $^{{***}}p{{<}}0.001$."
                )
                lines.append(r"\begin{table*}[!htb]")
                lines.append(r"  \centering")
                lines.append(r"  \small")
                lines += [
                    r"  \caption{%",
                    f"    {cap}%",
                    r"  }",
                    rf"  \label{{{label}}}",
                ]
                if _tbl_exists(stem):
                    lines.append(rf"  \input{{tables/{stem}}}")
                else:
                    lines.append(f"  % MISSING: tables/{stem}.tex")
                lines += [r"\end{table*}", ""]

        lines.append(r"\clearpage")
        
        # ── 2.x.5  SSMD and Z'-factor metrics ───────────────────────────
        # One 9-panel figure per metric (SSMD, Z'-factor).
        # Replicates the layout of screening_data_ssmd.tex and
        # screening_data_z_factor.tex: 3 rows × 3 cols, each 0.3\textwidth.
        # Only the 10-10 control configuration is shown (matches the wrappers).
        # Filename pattern written by util.plotting_residual_metrics:
        #   figures/screening-{metric}-mse-{csv_basename}.png
        lines += [r"\subsubsection{SSMD and Z$'$-factor metrics}", ""]

        neg_m, pos_m = _METRICS_DISPLAY_CONTROLS
        ctrl_desc_m = _CTRL_LABEL.get((neg_m, pos_m), f"{pos_m} -- {neg_m} controls")

        for metric_key, metric_tex in [("SSMD", "SSMD"), ("Zfactor", "Z$'$-factor")]:
            label = f"fig:screening-{metric_key.lower()}-{pos_m}-{neg_m}"
            cap = (
                rf"Comparison of MSE when calculating the {metric_tex} of plates "
                rf"for \emph{{{emph}}} plate effects "
                rf"using {ctrl_desc_m} on a 384-well plate with 1~replicate, "
                rf"1\%~hit-rate and varying strengths of bowl-shaped plate effects. "
                rf"$^{{*}}p{{<}}0.05$, $^{{**}}p{{<}}0.01$, $^{{***}}p{{<}}0.001$."
            )

            lines.append(r"\begin{figure*}[!htb]")
            lines.append(r"  \centering")

            errors = _METRICS_DISPLAY_ERRORS  # 9 items, 3×3 grid
            for i, error in enumerate(errors):
                csv_basename = (
                    f"screening_metrics_data-{neg_m}-{pos_m}-{error}"
                    f"-{cfg.metrics_date_tag}-{d.key}-{cfg.metrics_id_text}.csv"
                )
                png = f"screening-{metric_key}-mse-{csv_basename}.png"
                sub_cap = "No plate effect" if error == 0.0 else str(i)
                lines += [
                    rf"  \begin{{subfigure}}[b]{{0.3\textwidth}}",
                    r"    \centering",
                    rf"    \caption{{{sub_cap}}}",
                    rf"    \includegraphics[width=\textwidth]{{figures/{png}}}" if _fig_exists(png) else f"    % MISSING: figures/{png}",
                    r"  \end{subfigure}",
                ]
                # \hfill after cols 1 and 2 of each row (not after col 3)
                if i % 3 < 2:
                    lines.append(r"  \hfill")
                elif i % 3 == 2 and i < len(errors) - 1:
                    lines.append("")  # blank line between rows

            lines += [
                rf"  \caption{{{cap}}}",
                rf"  \label{{{label}}}",
                r"\end{figure*}",
                "",
            ]

        lines.append(r"\clearpage")

    out_path.write_text("\n".join(lines) + "\n")
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
        choices=["simulate", "figures", "metrics", "tables", "all"],
        default="all",
        help=(
            "simulate: generate CSVs; "
            "figures:  generate screening/ROC/PR plots; "
            "metrics:  generate SSMD/Z' plots; "
            "tables:   generate LaTeX AUC tables; "
            "all:      from simulate to figures to metrics to tables"
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
    if args.stage in ("tables", "all"):
        generate_auc_latex_tables(cfg)
        generate_auc_overview_tables(cfg)
        generate_metrics_latex_tables(cfg)
        generate_screening_section_tex(cfg)


if __name__ == "__main__":
    main()
