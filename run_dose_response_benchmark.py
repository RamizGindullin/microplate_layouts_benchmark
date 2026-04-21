#!/usr/bin/env python3
"""
run_dose_response_benchmark.py

Consolidated driver for the dose–response side of the COMPD vs PLAID benchmark.

Replaces:
  - dose-response-experiments.ipynb  (data generation)
  - dose-response-supplement.ipynb   (dose–response residual/IC50/d_max/R2 figures)
  - parts of dose-response-curves.ipynb (example curve PNGs), once curves stage is completed.

The script is structured in "stages", mirroring run_screening_benchmark.py:
  - simulate : generate CSVs under generated-data/dose-response/
  - figures  : generate dose–response figures under generated-plots/dose-response-supplement/
               (and some paper panels under figures/)
  - curves   : (optional) generate example per-compound curves in figures/
  - all      : run simulate, then figures (and curves when enabled)

It is designed to preserve existing filenames and paths so that LaTeX sources compile without changes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import libraries.disturbances as dt
import libraries.normalization as nrm
import libraries.dose_response as dr
import libraries.utilities as util


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class DoseResponseScenario:
    """
    Encodes a single disturbance scenario from id_text_error_level_type_list
    in dose-response-experiments.ipynb.

    id_text and error_nl together determine the CSV filename suffixes.
    error_types is the exact list of dicts passed into full_dose_response_evaluation.
    label is the suffix used in figure names (see dose-response-supplement.ipynb).
    """
    id_text: str
    error_nl: float
    error_types: List[Dict[str, Any]]
    label: str


@dataclass
class DoseResponseConfig:
    """
    Central configuration for the dose–response benchmark.

    Paths and constants are chosen to match the current notebooks and LaTeX.
    """

    base_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(
        default_factory=lambda: Path("generated-data") / "dose-response"
    )
    figures_dir: Path = field(
        default_factory=lambda: Path("generated-plots") / "dose-response-supplement"
    )
    paper_figures_dir: Path = field(default_factory=lambda: Path("figures"))

    concentrations_list: List[int] = field(default_factory=lambda: [6, 8, 12])
    replicates_list: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Fixed tag to match the 20250706-* filenames currently hardcoded in
    # dose-response-supplement.ipynb and the LaTeX sources.
    date_tag: str = "20250706-"

    # Scenarios cloned from id_text_error_level_type_list in dose-response-experiments.ipynb.
    # Labels are chosen to match fig_name suffixes in the supplement.
    scenarios: List[DoseResponseScenario] = field(
        default_factory=lambda: [
            DoseResponseScenario(
                id_text="right-half-neg-control-log-new-reg",
                error_nl=0.2,
                error_types=[
                    {
                        "type": "right-half",
                        "error_function": dt.add_errors_to_right_columns_half,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.2,
                    }
                ],
                label="half-columns-neg-controls-0.2",
            ),
            DoseResponseScenario(
                id_text="log-neg-control-new-reg",
                error_nl=0.4,
                error_types=[
                    {
                        "type": "right-half",
                        "error_function": dt.add_errors_to_right_columns_half,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.4,
                    }
                ],
                # This scenario is mostly used for robustness runs / grids; no direct figures.
                label="half-columns-neg-controls-0.4-log",
            ),
            DoseResponseScenario(
                id_text="right-half-neg-control-log-new-reg",
                error_nl=0.4,
                error_types=[
                    {
                        "type": "right-half",
                        "error_function": dt.add_errors_to_right_columns_half,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.4,
                    }
                ],
                label="half-columns-neg-controls-0.4",
            ),
            DoseResponseScenario(
                id_text="curve_info-new-reg",
                error_nl=0.055,
                error_types=[
                    {
                        "type": "bowl-nl",
                        "error_function": dt.add_bowlshaped_errors_nl,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.055,
                    }
                ],
                label="bowl-0.055",
            ),
            DoseResponseScenario(
                id_text="bowl-neg-control-new-reg",
                error_nl=0.055,
                error_types=[
                    {
                        "type": "bowl-nl",
                        "error_function": dt.add_bowlshaped_errors_nl,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.055,
                    }
                ],
                label="bowl-neg-controls-0.055",
            ),
            DoseResponseScenario(
                id_text="curve_info-new-reg",
                error_nl=0.085,
                error_types=[
                    {
                        "type": "bowl-nl",
                        "error_function": dt.add_bowlshaped_errors_nl,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.085,
                    }
                ],
                label="bowl-0.085",
            ),
            DoseResponseScenario(
                id_text="bowl-neg-control-new-reg",
                error_nl=0.085,
                error_types=[
                    {
                        "type": "bowl-nl",
                        "error_function": dt.add_bowlshaped_errors_nl,
                        "error_correction": nrm.normalize_plate_nearest_control,
                        "error": 0.085,
                    }
                ],
                label="bowl-neg-controls-0.085",
            ),
        ]
    )

    def plate_types_location(
        self, compounds: int, concentrations: int, replicates: int
    ) -> List[Dict[str, Any]]:
        """
        Recreates plate_types_location from the dose-response-experiments notebook
        for a given (compounds, concentrations, replicates) triple.
        """
        return [
            {
                "type": "COMPD",
                "dir": "layouts/compounds_COMPD_layouts/",
                "regex": f"plate_layout_(.*){compounds}-{concentrations}-{replicates}_(0*)(.+?).npy",
                "error_correction": nrm.normalize_plate_lowess_2d,
            },
            {
                "type": "PLAID",
                "dir": "layouts/compounds_PLAID_layouts/",
                "regex": f"plate_layout_(.*){compounds}-{concentrations}-{replicates}_(0*)(.+?).npy",
                "error_correction": nrm.normalize_plate_lowess_2d,
            },
            {
                "type": "RANDOM",
                "dir": "layouts/compounds_manual_layouts/",
                "regex": "plate_layout_rand_(.+?).npy",
                "error_correction": nrm.normalize_plate_lowess_2d,
            },
        ]


# ----------------------------------------------------------------------
# Stage 1: Simulations (CSV generation)
# ----------------------------------------------------------------------


def run_simulations(cfg: DoseResponseConfig) -> None:
    """
    Reproduces the driver loop in dose-response-experiments.ipynb, calling
    full_dose_response_evaluation for all (concentrations, replicates, scenario)
    combinations used by the supplement and main paper.

    This writes absolute_ic50_data-*, relative_ic50_data-* and residuals-* CSVs
    into cfg.data_dir, with filenames exactly matching the notebooks.
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    r = 0
    r_max = len(cfg.concentrations_list) * len(cfg.replicates_list) * len(cfg.scenarios)

    for concentrations in cfg.concentrations_list:
        for replicates in cfg.replicates_list:
            # This reproduces the original notebook calculation.
            compounds = (14 * 22 - 20) // (concentrations * replicates)

            if concentrations == 4:
                dilution = 15
            elif concentrations == 6:
                dilution = 18
            elif concentrations == 8:
                dilution = 8
            elif concentrations == 12:
                dilution = 4
            else:
                raise ValueError(f"Unsupported concentrations: {concentrations}")

            for scenario in cfg.scenarios:
                today = cfg.date_tag  # e.g. "20250706-"

                print(
                    "Results are being stored in files which include the name:",
                    f"{compounds}-{concentrations}-dil{dilution}-{replicates}-"
                    f"{scenario.error_nl}-{today}{scenario.id_text}",
                )

                plate_types_location = cfg.plate_types_location(
                    compounds, concentrations, replicates
                )

                dr.full_dose_response_evaluation(
                    plate_types_location,
                    scenario.error_types,
                    compounds=compounds,
                    concentrations=concentrations,
                    replicates=replicates,
                    dilution=dilution,
                    error_nl=scenario.error_nl,
                    today=today + scenario.id_text,
                    data_dir=str(cfg.data_dir) + os.sep,
                )
                r += 1
                print(r, "out of", r_max)


# ----------------------------------------------------------------------
# Helpers for figure generation
# ----------------------------------------------------------------------


def _compounds_for(doses: int, replicates: int) -> int:
    """
    Helper to reconstruct the compounds count for a given doses/replicate
    configuration, mirroring the notebook.
    """
    return (14 * 22 - 20) // (doses * replicates)


def _residuals_paths(
    cfg: DoseResponseConfig, doses: int, dilution: int, error_nl: float, id_text: str
) -> List[Path]:
    """
    Build the three residuals CSV filenames for 1,2,3 replicates, as used in
    dose-response-supplement.ipynb for residual MSE panels.
    """
    comps = [
        _compounds_for(doses, 1),
        _compounds_for(doses, 2),
        _compounds_for(doses, 3),
    ]
    reps = [1, 2, 3]
    paths: List[Path] = []
    for c, r in zip(comps, reps):
        fname = (
            f"residuals-{c}-{doses}-dil{dilution}-{r}-{error_nl}-"
            f"{cfg.date_tag}{id_text}.csv"
        )
        paths.append(cfg.data_dir / fname)
    return paths


def _load_residuals_triple(
    cfg: DoseResponseConfig, doses: int, dilution: int, error_nl: float, id_text: str
) -> List[np.ndarray]:
    paths = _residuals_paths(cfg, doses, dilution, error_nl, id_text)
    return [np.loadtxt(p, delimiter=",", dtype="str") for p in paths]

def _ic50_paths(
    cfg: DoseResponseConfig,
    metric: str,          # "absolute" or "relative"
    doses: int,
    dilution: int,
    error_nl: float,
    id_text: str,
) -> List[Path]:
    """
    Build the three absolute/relative IC50 CSV filenames (1,2,3 reps), matching
    dose-response-experiments.ipynb and the artifact map.

    metric == "absolute" → absolute_ic50_data-...
    metric == "relative" → relative_ic50_data-...
    """
    if metric == "absolute":
        prefix = "absolute_ic50_data"
    elif metric == "relative":
        prefix = "relative_ic50_data"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    comps = [
        _compounds_for(doses, 1),
        _compounds_for(doses, 2),
        _compounds_for(doses, 3),
    ]
    reps = [1, 2, 3]

    paths: List[Path] = []
    for c, r in zip(comps, reps):
        fname = (
            f"{prefix}-{c}-{doses}-dil{dilution}-{r}-{error_nl}-"
            f"{cfg.date_tag}{id_text}.csv"
        )
        paths.append(cfg.data_dir / fname)
    return paths

def _load_ic50_triple(
    cfg: DoseResponseConfig,
    metric: str,
    doses: int,
    dilution: int,
    error_nl: float,
    id_text: str,
) -> List[np.ndarray]:
    """
    Load the 1-, 2-, 3-replicate CSVs for the given metric and scenario as
    np.ndarray objects suitable for plot_barplot_replicate_data / plot_r2_percentage.
    """
    paths = _ic50_paths(cfg, metric, doses, dilution, error_nl, id_text)
    return [np.loadtxt(p, delimiter=",", dtype="str") for p in paths]


# ----------------------------------------------------------------------
# Stage 2: Figures (residuals + IC50/d_max/R2)
# ----------------------------------------------------------------------


def generate_residuals_figures(cfg: DoseResponseConfig) -> None:
    """
    Regenerates all residual MSE panels from the supplement (Group 3) plus the
    special 8-dose paper panel.

    Output filenames match the artifact map exactly, e.g.:
      residuals-1-2-3-8doses-dil8-bowl-0.055.png
      residuals-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png
    """
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # Bowl-shaped, no negatives in the fit (curve_info-new-reg)
    for doses, dilution in [(6, 18), (8, 8), (12, 4)]:
        for error_nl in (0.055, 0.085):
            residuals_1rep, residuals_2rep, residuals_3rep = _load_residuals_triple(
                cfg,
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="curve_info-new-reg",
            )
            fig_name = f"-1-2-3-{doses}doses-dil{dilution}-bowl-{error_nl}"
            util.plot_barplot_residuals_data(
                residuals_1rep,
                residuals_2rep,
                residuals_3rep,
                fig_name,
                y_max=450,
                leg_loc="upper center",
                fig_dir=str(cfg.figures_dir) + os.sep,
            )

    # Bowl-shaped, with 4 negatives in the fit (bowl-neg-control-new-reg)
    for doses, dilution in [(6, 18), (8, 8), (12, 4)]:
        for error_nl in (0.055, 0.085):
            residuals_1rep, residuals_2rep, residuals_3rep = _load_residuals_triple(
                cfg,
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="bowl-neg-control-new-reg",
            )
            fig_name = (
                f"-1-2-3-{doses}doses-dil{dilution}-bowl-neg-controls-{error_nl}"
            )
            util.plot_barplot_residuals_data(
                residuals_1rep,
                residuals_2rep,
                residuals_3rep,
                fig_name,
                y_max=450,
                leg_loc="upper center",
                fig_dir=str(cfg.figures_dir) + os.sep,
            )

    # Column-wise right-half effects with 4 negatives (0.2 / 0.4)
    for doses, dilution in [(6, 18), (8, 8), (12, 4)]:
        for error_nl in (0.2, 0.4):
            residuals_1rep, residuals_2rep, residuals_3rep = _load_residuals_triple(
                cfg,
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="right-half-neg-control-log-new-reg",
            )
            fig_name = (
                f"-1-2-3-{doses}doses-dil{dilution}-half-columns-neg-controls-{error_nl}"
            )
            util.plot_barplot_residuals_data(
                residuals_1rep,
                residuals_2rep,
                residuals_3rep,
                fig_name,
                y_max=450,
                leg_loc="upper center",
                fig_dir=str(cfg.figures_dir) + os.sep,
            )

    # Special paper residual panel (8 doses, strong column disturbance with 4 negatives)
    doses, dilution, error_nl = 8, 8, 0.4
    residuals_1rep, residuals_2rep, residuals_3rep = _load_residuals_triple(
        cfg,
        doses=doses,
        dilution=dilution,
        error_nl=error_nl,
        id_text="right-half-neg-control-log-new-reg",
    )
    fig_name = "-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper"
    util.plot_barplot_residuals_data(
        residuals_1rep,
        residuals_2rep,
        residuals_3rep,
        fig_name,
        y_max=450,
        fig_dir=str(cfg.figures_dir) + os.sep,
    )


def generate_ic50_dmax_r2_figures(cfg: DoseResponseConfig) -> None:
    """
    Regenerates all Group 2 and Group 3 dose–response figures that depend on
    IC50/EC50 errors, d_max differences, and fit-quality (R²) using the
    CSVs written by run_simulations().

    This covers:
      - dose-response-d_diff-... (Groups 2 + paper row)
      - dose-response-relic50-... (Group 3 + paper row)
      - dose-response-absic50-... (Group 3 + paper row)
      - percentage-low-r2-curves-1-2-3-... (Group 3)
    """
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Bowl-shaped, no negatives in fit (curve_info-new-reg) ---
    for doses, dilution in [(6, 18), (8, 8), (12, 4)]:
        for error_nl in (0.055, 0.085):
            # Relative and absolute IC50 CSVs
            rel_1, rel_2, rel_3 = _load_ic50_triple(
                cfg,
                metric="relative",
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="curve_info-new-reg",
            )
            abs_1, abs_2, abs_3 = _load_ic50_triple(
                cfg,
                metric="absolute",
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="curve_info-new-reg",
            )

            # d_max (diff_d uses d, fit_d inside the data)
            fig_name = f"-1-2-3-{doses}doses-dil{dilution}-bowl-{error_nl}"
            util.plot_barplot_replicate_data(
                abs_1,
                abs_2,
                abs_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="",  # triggers diff_d and fig_type="d_diff"
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Relative IC50/EC50
            util.plot_barplot_replicate_data(
                rel_1,
                rel_2,
                rel_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="relic50",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Absolute IC50/EC50
            util.plot_barplot_replicate_data(
                abs_1,
                abs_2,
                abs_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="absic50",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Percentage of low-R² curves (R² < 0.8)
            r2_fig_name = f"-{doses}doses-dil{dilution}-bowl-{error_nl}"
            util.plot_r2_percentage(
                rel_1,
                rel_2,
                rel_3,
                fig_name=r2_fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                leg_loc="upper left",
                leg_ncol=1,
                leg_fontsize=8,
            )

    # --- Bowl-shaped, with 4 negatives in fit (bowl-neg-control-new-reg) ---
    for doses, dilution in [(6, 18), (8, 8), (12, 4)]:
        for error_nl in (0.055, 0.085):
            rel_1, rel_2, rel_3 = _load_ic50_triple(
                cfg,
                metric="relative",
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="bowl-neg-control-new-reg",
            )
            abs_1, abs_2, abs_3 = _load_ic50_triple(
                cfg,
                metric="absolute",
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="bowl-neg-control-new-reg",
            )

            fig_name = (
                f"-1-2-3-{doses}doses-dil{dilution}-bowl-neg-controls-{error_nl}"
            )

            # d_max
            util.plot_barplot_replicate_data(
                abs_1,
                abs_2,
                abs_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Relative IC50/EC50
            util.plot_barplot_replicate_data(
                rel_1,
                rel_2,
                rel_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="relic50",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Absolute IC50/EC50
            util.plot_barplot_replicate_data(
                abs_1,
                abs_2,
                abs_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="absic50",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # R² failure rate
            r2_fig_name = (
                f"-{doses}doses-dil{dilution}-bowl-neg-controls-{error_nl}"
            )
            util.plot_r2_percentage(
                rel_1,
                rel_2,
                rel_3,
                fig_name=r2_fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                leg_loc="upper left",
                leg_ncol=1,
                leg_fontsize=8,
            )

    # --- Column-wise right-half effects with 4 negatives (0.2 / 0.4) ---
    for doses, dilution in [(6, 18), (8, 8), (12, 4)]:
        for error_nl in (0.2, 0.4):
            rel_1, rel_2, rel_3 = _load_ic50_triple(
                cfg,
                metric="relative",
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="right-half-neg-control-log-new-reg",
            )
            abs_1, abs_2, abs_3 = _load_ic50_triple(
                cfg,
                metric="absolute",
                doses=doses,
                dilution=dilution,
                error_nl=error_nl,
                id_text="right-half-neg-control-log-new-reg",
            )

            fig_name = (
                f"-1-2-3-{doses}doses-dil{dilution}-half-columns-neg-controls-{error_nl}"
            )

            # d_max
            util.plot_barplot_replicate_data(
                abs_1,
                abs_2,
                abs_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Relative IC50/EC50
            util.plot_barplot_replicate_data(
                rel_1,
                rel_2,
                rel_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="relic50",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # Absolute IC50/EC50
            util.plot_barplot_replicate_data(
                abs_1,
                abs_2,
                abs_3,
                fig_name=fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                fig_type="absic50",
                leg_loc="upper center",
                leg_ncol=3,
                leg_fontsize=8,
            )

            # R² failure rate
            r2_fig_name = (
                f"-{doses}doses-dil{dilution}-half-columns-neg-controls-{error_nl}"
            )
            util.plot_r2_percentage(
                rel_1,
                rel_2,
                rel_3,
                fig_name=r2_fig_name,
                fig_dir=str(cfg.figures_dir) + os.sep,
                leg_loc="upper left",
                leg_ncol=1,
                leg_fontsize=8,
            )

    # --- Paper IC50/d_max panels (8 doses, 0.4, half-columns-neg-controls, _paper) ---
    doses, dilution, error_nl = 8, 8, 0.4
    rel_1, rel_2, rel_3 = _load_ic50_triple(
        cfg,
        metric="relative",
        doses=doses,
        dilution=dilution,
        error_nl=error_nl,
        id_text="right-half-neg-control-log-new-reg",
    )
    abs_1, abs_2, abs_3 = _load_ic50_triple(
        cfg,
        metric="absolute",
        doses=doses,
        dilution=dilution,
        error_nl=error_nl,
        id_text="right-half-neg-control-log-new-reg",
    )
    paper_fig_name = "-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper"

    # d_max paper panel
    util.plot_barplot_replicate_data(
        abs_1,
        abs_2,
        abs_3,
        fig_name=paper_fig_name,
        fig_dir=str(cfg.figures_dir) + os.sep,
        fig_type="",
        leg_loc="upper center",
        leg_ncol=3,
        leg_fontsize=8,
    )

    # Relative IC50/EC50 paper panel
    util.plot_barplot_replicate_data(
        rel_1,
        rel_2,
        rel_3,
        fig_name=paper_fig_name,
        fig_dir=str(cfg.figures_dir) + os.sep,
        fig_type="relic50",
        leg_loc="upper center",
        leg_ncol=3,
        leg_fontsize=8,
    )

    # Absolute IC50/EC50 paper panel
    util.plot_barplot_replicate_data(
        abs_1,
        abs_2,
        abs_3,
        fig_name=paper_fig_name,
        fig_dir=str(cfg.figures_dir) + os.sep,
        fig_type="absic50",
        leg_loc="upper center",
        leg_ncol=3,
        leg_fontsize=8,
    )


def generate_dose_response_figures(cfg: DoseResponseConfig) -> None:
    """
    Orchestrates all dose–response figure generation.

    After this runs, all Group 1–3 dose–response PNGs listed in the artifact
    map are regenerated from the CSVs produced by run_simulations().
    """
    generate_residuals_figures(cfg)
    generate_ic50_dmax_r2_figures(cfg)


# ----------------------------------------------------------------------
# Stage 3: Curves (example per-compound PNGs)
# ----------------------------------------------------------------------


def generate_example_curves(cfg: DoseResponseConfig) -> None:
    """
    Repackages the core of dose-response-curves.ipynb to regenerate example
    per-compound curve PNGs like:

      figures/plate_layout_rand_02.npy_compound_1-right-half.png
      figures/plate_layout_20-12-8-3_01.npy_compound_1-right-half.png
      figures/plate_layout_40-12-8-3_01.npy_compound_1-right-half.png
      ...

    This function is intentionally conservative: it only reproduces the layouts
    and scenarios actually used in the paper/supplement artifact map. You can
    extend it if you later want all debug PNGs.
    """
    cfg.paper_figures_dir.mkdir(parents=True, exist_ok=True)

    # Example: use a single representative e-range around 50, as in the curves notebook.
    concentrations = 8
    replicates = 3
    dilution = 8

    # Number of compounds chosen to match the 12x8 grids in the example layouts
    compounds = 40

    # Slopes from the curves notebook
    slopes = [0.5, 1, 1.5, 2]

    # Disturbance: right-half with 4 negatives, strength 0.4 (paper scenario)
    error_nl = 0.4
    error_types = [
        {
            "type": "right-half",
            "error_function": dt.add_errors_to_right_columns_half,
            "error_correction": nrm.normalize_plate_nearest_control,
            "error": error_nl,
        }
    ]

    # Build df_params / plate_content similarly to dose-response-curves.ipynb
    params = []
    current_e = 50
    for i in range(compounds):
        params.append(
            {
                "compound": i,
                "b": slopes[i % 3],
                "c": 0,
                "d": 100,
                "e": current_e + 5 * np.random.random(),
                "startDose": 10000,
                "nDose": concentrations,
                "dilution": dilution,
            }
        )

    df_params = pd.DataFrame.from_dict(params)
    df_params.set_index("compound", inplace=True)
    df_params["abs IC50"] = dr.IC50(
        df_params["b"], df_params["c"], df_params["d"], df_params["e"]
    )

    plate_content = dr.generate_plate_content(
        dose_response_params=params, replicates=replicates
    )

    # Layout sets taken from dose-response-curves.ipynb
    plate_types_location = [
        {
            "type": "COMPD",
            "dir": "layouts/compounds_COMPD_layouts/",
            "layouts": [
                "plate_layout_40-12-8-3_01.npy",
            ],
        },
        {
            "type": "PLAID",
            "dir": "layouts/compounds_PLAID_layouts/",
            "layouts": [
                "plate_layout_20-12-8-3_01.npy",
            ],
        },
        {
            "type": "Random",
            "dir": "layouts/compounds_manual_layouts/",
            "layouts": [
                "plate_layout_rand_02.npy",
            ],
        },
    ]

    expected_noise = 0.01
    my_min_dist = 0  # as in the curves notebook

    # Ensure we write PNGs into the 'figures/' directory
    cwd = os.getcwd()
    os.chdir(cfg.paper_figures_dir)

    try:
        for plate_type in plate_types_location:
            layout_dir = plate_type["dir"]
            for layout_file in plate_type["layouts"]:
                for et in error_types:
                    # We only need one lost_rows / limit here to trigger plotting.
                    limits = [{"from": 15, "to": 16}]  # bottom row, as in notebooks
                    for limit in limits:
                        dr.plate_curves_after_error(
                            layout_dir,
                            layout_file,
                            plate_content,
                            expected_noise,
                            et["error_function"],
                            et["error"],
                            plate_type.get("error_correction", nrm.normalize_plate_lowess_2d),
                            my_min_dist,
                            lose_from_row=limit["from"],
                            lose_to_row=limit["to"],
                            df_params=df_params,
                            plate_type=plate_type,
                            compounds=compounds,
                            concentrations=concentrations,
                            replicates=replicates,
                        )
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run COMPD/PLAID dose–response benchmark pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["simulate", "figures", "curves", "all"],
        default="all",
        help="Which part of the pipeline to run.",
    )
    args = parser.parse_args()

    cfg = DoseResponseConfig()

    if args.stage in ("simulate", "all"):
        run_simulations(cfg)
    if args.stage in ("figures", "all"):
        generate_dose_response_figures(cfg)
    if args.stage in ("curves", "all"):
        generate_example_curves(cfg)


if __name__ == "__main__":
    main()