#!/usr/bin/env python3
"""
run_dose_response_benchmark.py

Consolidated driver for the dose–response side of the COMPD vs PLAID benchmark.

Replaces:
  - dose-response-experiments.ipynb  (data generation)
  - dose-response-supplement.ipynb   (dose–response residual/IC50/d_max/R2 figures)
  - dose-response-curves.ipynb       (example curve PNGs)

Stages:
  simulate : generate CSVs under generated-data/dose-response/
  figures  : generate supplement PNGs
  curves   : generate example curve PNGs
  tables   : generate LaTeX tables
  all      : simulate to figures to tables to curves  (default)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
np.random.seed(42)
import pandas as pd  # used in generate_example_curves (df_params)

import libraries.disturbances as dt
import libraries.normalization as nrm
import libraries.dose_response as dr
import libraries.utilities as util
from benchmark_common import (
    DOSE_RESPONSE_FIGURE_CASES,
    DOSE_RESPONSE_LAYOUT_SPECS,
    DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER,
    dose_response_curve_examples,
    dose_response_plate_types,
    dilution_for,
    fig_dir_str,
    validate_layout_registry_consistency,
    dr_curve_stem,
    dr_curve_png_name
)
from benchmark_disturbances import (
    DISTURBANCES,
    dr_scenarios,
    DR_LABEL_BY_ID,
    DR_FILE_SUFFIX_BY_ID,
    DR_STEM_LABEL_BY_ID,
    _disturbance_function_for_dr_id
)
import libraries.disturbances as _dt_for_registry


CURVE_FIGURE_SCENARIO_KEY = "right-half-s0.4"
CURVE_FIGURE_COMPOUND = 1
CURVE_FIGURE_TEX_NAME = "dose_response_illustrative_curves.tex"
CURVE_FIGURE_COMPOUNDS = None       # All compounds, default
# CURVE_FIGURE_COMPOUNDS = [0, 1, 2]  # Explicit subset

# ── Caption templates (verbatim style from existing thin wrappers) ─────────
_DR_CAPTION_PLACEHOLDER = "DISTURBANCE_DESC"

_DR_CAPTIONS = {
    "d_diff": (
        r"Mean absolute difference between expected and obtained heights "
        r"($d_{\max}$) for dose-response curves using various numbers of "
        r"doses, replicates, and " + _DR_CAPTION_PLACEHOLDER + r". "
        r"The 4PL sigmoid curves were fitted using compound data and 4 negative "
        r"controls such that their mean was equal to the mean of the 20 "
        r"negative controls on the plate. *** indicates $p\leq10^{-43}$."
    ),
    "relic50": (
        r"Relative IC$_{50}$/EC$_{50}$ error for dose-response curves "
        r"using various numbers of doses, replicates, and " + _DR_CAPTION_PLACEHOLDER + r". "
        r"*** indicates $p\leq10^{-43}$."
    ),
    "absic50": (
        r"Absolute IC$_{50}$/EC$_{50}$ error for dose-response curves "
        r"using various numbers of doses, replicates, and " + _DR_CAPTION_PLACEHOLDER + r". "
        r"*** indicates $p\leq10^{-43}$."
    ),
    "residuals": (
        r"Residuals (MSE between expected and obtained responses) for "
        r"dose-response simulations using various numbers of doses, replicates, "
        r"and " + _DR_CAPTION_PLACEHOLDER + r". *** indicates $p\leq10^{-43}$."
    ),
    "percentage": (
        r"Percentage of dose-response curves with low-quality "
        r"($R^2 < 80\%$) using various numbers of doses, replicates, and "
        + _DR_CAPTION_PLACEHOLDER + r". "
        r"The 4PL sigmoid curves were fitted using compound data and 4 negative "
        r"controls such that their mean was equal to the mean of the 20 "
        r"negative controls on the plate."
    ),
}

_DR_TABLE_CAPTIONS = {
    "dmax": (
        r"$d_{max}$ error overview for "
        + _DR_CAPTION_PLACEHOLDER
        + r". Values are mean absolute differences in fitted $d_{max}$ across compounds. Best (lowest) value per row is \textbf{bolded}."
    ),
    "rel-ic50": (
        r"Relative IC$_{50}$/EC$_{50}$ MSE overview for "
        + _DR_CAPTION_PLACEHOLDER
        + r". Best (lowest) value per row is \textbf{bolded}."
    ),
    "abs-ic50": (
        r"Absolute IC$_{50}$/EC$_{50}$ error overview for "
        + _DR_CAPTION_PLACEHOLDER
        + r". Values are mean squared absolute IC$_{50}$/EC$_{50}$ estimation errors across compounds. Best (lowest) value per row is \textbf{bolded}."
    ),
    "residuals": (
        r"Residual MSE overview for "
        + _DR_CAPTION_PLACEHOLDER
        + r". Best (lowest) value per row is \textbf{bolded}."
    ),
}

# ── Subsubsection titles ───────────────────────────────────────────────────
_DR_SUBSUBSECTION = {
    "d_diff":      r"$d_{\max}$ errors",
    "relic50":     r"Relative IC$_{50}$/EC$_{50}$ errors",
    "absic50":     r"Absolute IC$_{50}$/EC$_{50}$ errors",
    "residuals":   r"Residuals",
    "percentage":  r"Fraction of poor fits (low $R^2$)",
}

validate_layout_registry_consistency()

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_csv_triple(
    cfg: "DoseResponseConfig",
    prefix: str,
    doses: int,
    dilution: int,
    error_nl: float,
    id_text: str,
) -> List[np.ndarray]:
    """
    Load the 1-, 2-, 3-replicate CSVs for *prefix* (e.g. 'residuals',
    'relative_ic50_data', 'absolute_ic50_data') and the given scenario
    as np.ndarray objects.
    """
    result = []
    for r in (1, 2, 3):
        compounds = _compounds_for(doses, r)
        fname = (
            f"{prefix}-{compounds}-{doses}-dil{dilution}-{r}-{error_nl}-"
            f"{cfg.date_tag}{id_text}.csv"
        )
        fpath = cfg.data_dir / fname
        try:
            result.append(np.loadtxt(fpath, delimiter=",", dtype="str"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing CSV for prefix={prefix!r}, doses={doses}, "
                f"dilution={dilution}, replicates={r}, error_nl={error_nl}, "
                f"id_text={id_text!r}. Expected: {fpath}"
            ) from exc
    return result


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class DoseResponseScenario:
    """
    Encodes a single disturbance scenario.

    id_text + error_nl determine the CSV filename suffixes written by
    run_simulations and read by the figure-generation stage.
    Figure name suffixes are derived from IC50_DMAX_R2_SCENARIO_GROUPS lambdas.

    NOTE: error_correction lives inside each error_types dict entry here
    (not on the layout/PlateType), because dose-response normalisation is
    matched to the disturbance shape. This is the opposite convention from the
    screening path, where normalisation is per-layout. See benchmark_common.py
    SCREENING_LAYOUT_SPECS for the screening side.
    """
    id_text: str
    error_nl: float
    error_types: List[Dict[str, Any]]

def _build_default_dr_scenarios() -> List[DoseResponseScenario]:
    """
    Build the canonical list of DoseResponseScenario objects from the
    disturbance registry + benchmark_common error-level constants.

    This function is the single place that maps:
        registry dr_id_text  →  error_function + error_types dict structure

    The error_nl values come from benchmark_disturbances; the function
    callables are resolved by the registry helper.
    Behaviour is identical to the previous hardcoded list.
    """
    
    scenarios = []
    for d in dr_scenarios():
        id_text = d.dr_id_text
        fn = _disturbance_function_for_dr_id(id_text)
        for lv in d.dr_error_levels:
            error_nl = lv.value
            scenarios.append(
                DoseResponseScenario(
                    id_text=id_text,
                    error_nl=error_nl,
                    error_types=[{
                        "type":                     d.dr_error_type,
                        "neg_controls_affected":    d.dr_neg_controls_affected,
                        "error_function":           fn,
                        "error":                    error_nl,
                    }],
                )
            )
    return scenarios

@dataclass
class DoseResponseConfig:
    base_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(
        default_factory=lambda: Path("generated-data") / "dose-response"
    )
    figures_dir: Path = field(
        default_factory=lambda: Path("detailed-experimental-results-source") / "figures"
    )
    latex_tables_dir: Path = field(
        default_factory=lambda: Path("detailed-experimental-results-source") / "tables"
    )
    paper_figures_dir: Path = field(
        default_factory=lambda: Path("detailed-experimental-results-source") / "figures"
    )

    concentrations_list: List[int] = field(default_factory=lambda: [6, 8, 12])
    replicates_list: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Fixed tag matching the 20250706-* filenames in the LaTeX sources.
    date_tag: str = "20250706-"

    scenarios: List[DoseResponseScenario] = field(
        default_factory=_build_default_dr_scenarios
    )

    def plate_types_location(
        self, compounds: int, concentrations: int, replicates: int
    ) -> List[Dict[str, Any]]:
        return dose_response_plate_types(compounds, concentrations, replicates)


# -----------------------------------------------------------------------
# Stage 1: Simulations
# -----------------------------------------------------------------------

def _compounds_for(concentrations: int, replicates: int) -> int:
    """Reproduces the notebook formula: int((14*22 - 20) / (conc * rep))."""
    return (14 * 22 - 20) // (concentrations * replicates)


def run_simulations(cfg: DoseResponseConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    for concentrations in cfg.concentrations_list:
        for replicates in cfg.replicates_list:
            compounds = _compounds_for(concentrations, replicates)

            dilution = dilution_for(concentrations)

            for scenario in cfg.scenarios:
                today = cfg.date_tag

                print(
                    "Storing results with name:",
                    f"{compounds}-{concentrations}-dil{dilution}-{replicates}-"
                    f"{scenario.error_nl}-{today}{scenario.id_text}",
                )

                plate_types = cfg.plate_types_location(compounds, concentrations, replicates)

                dr.full_dose_response_evaluation(
                    plate_types,
                    scenario.error_types,
                    compounds=compounds,
                    concentrations=concentrations,
                    replicates=replicates,
                    dilution=dilution,
                    error_nl=scenario.error_nl,
                    today=today,
                    id_text=scenario.id_text,
                    data_directory=str(cfg.data_dir) + os.sep,
                )


RESIDUALS_SCENARIO_GROUPS = [
    (
        d.dr_id_text,
        tuple(lv.value for lv in d.dr_error_levels),
        lambda doses, dil, enl, _d=d: f"residuals-1-2-3-{doses}doses-dil{dil}-{_d.dr_stem_label}-{enl}",
    )
    for d in dr_scenarios()
]


# Each tuple: (id_text, error_levels, ic50_fig_name_fn, r2_fig_name_fn)
# ic50_fig_name_fn is used for d_diff / relic50 / absic50
#   filename: dose-response-{fig_type}{ic50_fig_name_fn(...)}.png
# r2_fig_name_fn is used for percentage-low-r2
#   filename: percentage-low-r2{r2_fig_name_fn(...)}.png
IC50_DMAX_R2_SCENARIO_GROUPS = [
    (
        d.dr_id_text,
        tuple(lv.value for lv in d.dr_error_levels),
        lambda doses, dil, enl, _d=d: f"-1-2-3-{doses}doses-dil{dil}-{_d.dr_stem_label}-{enl}",
        lambda doses, dil, enl, _d=d: f"-curves-1-2-3-{doses}doses-dil{dil}-{_d.dr_stem_label}-{enl}",
    )
    for d in dr_scenarios()
]

# -----------------------------------------------------------------------
# Stage 2: Figures
# -----------------------------------------------------------------------

def generate_residuals_figures(cfg: DoseResponseConfig) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = fig_dir_str(cfg.figures_dir)

    for id_text, error_nls, fig_name_fn in RESIDUALS_SCENARIO_GROUPS:
        for doses, dilution in DOSE_RESPONSE_FIGURE_CASES:
            for error_nl in error_nls:
                r1, r2, r3 = _load_csv_triple(
                    cfg, "residuals", doses, dilution, error_nl, id_text
                )
                util.plot_barplot_residuals_data(
                    r1, r2, r3,
                    fig_name=fig_name_fn(doses, dilution, error_nl),
                    y_max=450,
                    fig_dir=fig_dir,
                )

    # Special paper residual panel
    r1, r2, r3 = _load_csv_triple(
        cfg, "residuals", 8, 8, 0.4, "right-half-neg-control-log-new-reg"
    )
    util.plot_barplot_residuals_data(
        r1, r2, r3,
        fig_name="residuals-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper",
        y_max=450,
        fig_dir=fig_dir,
    )


def generate_ic50_dmax_r2_figures(cfg: DoseResponseConfig) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = fig_dir_str(cfg.figures_dir)

    for id_text, error_nls, fig_name_fn, r2_fig_name_fn in IC50_DMAX_R2_SCENARIO_GROUPS:
        for doses, dilution in DOSE_RESPONSE_FIGURE_CASES:
            for error_nl in error_nls:
                rel_1, rel_2, rel_3 = _load_csv_triple(
                    cfg, "relative_ic50_data", doses, dilution, error_nl, id_text
                )
                abs_1, abs_2, abs_3 = _load_csv_triple(
                    cfg, "absolute_ic50_data", doses, dilution, error_nl, id_text
                )
                fig_name = fig_name_fn(doses, dilution, error_nl)
                common_kwargs = dict(
                    fig_name=fig_name,
                    fig_dir=fig_dir,
                    leg_ncol=3,
                    leg_fontsize=8,
                )

                # d_max
                util.plot_barplot_replicate_data(
                    abs_1, abs_2, abs_3, fig_type="d_diff", **common_kwargs
                )
                # Relative IC50
                util.plot_barplot_replicate_data(
                    rel_1, rel_2, rel_3, fig_type="relic50", **common_kwargs
                )
                # Absolute IC50
                util.plot_barplot_replicate_data(
                    abs_1, abs_2, abs_3, fig_type="absic50", **common_kwargs
                )
                # R² failure rate (not applicable for right-half scenarios)
                if r2_fig_name_fn is not None:
                    util.plot_r2_percentage(
                        rel_1, rel_2, rel_3,
                        fig_name=r2_fig_name_fn(doses, dilution, error_nl),
                        fig_dir=fig_dir,
                        leg_ncol=1,
                        leg_fontsize=8,
                    )

    # Paper IC50/R² panels (8-dose, strong right-half)
    rel_1, rel_2, rel_3 = _load_csv_triple(
        cfg, "relative_ic50_data", 8, 8, 0.4, "right-half-neg-control-log-new-reg"
    )
    abs_1, abs_2, abs_3 = _load_csv_triple(
        cfg, "absolute_ic50_data", 8, 8, 0.4, "right-half-neg-control-log-new-reg"
    )
    # d_diff and absic50 use "right-half" in the filename; relic50 keeps "half-columns".
    # These two substrings must NOT be unified — they match different naming conventions
    # in the LaTeX sources and must correspond exactly to the filenames referenced there.
    _paper_common = dict(fig_dir=fig_dir, leg_loc="upper center", leg_ncol=3, leg_fontsize=8)
    # - dose-response-d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png
    util.plot_barplot_replicate_data(
        abs_1, abs_2, abs_3, fig_type="",
        fig_name="d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper",
        **_paper_common,
    )
    # - dose-response-relic50-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png
    util.plot_barplot_replicate_data(
        rel_1, rel_2, rel_3, fig_type="relic50",
        fig_name="-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper",
        **_paper_common,
    )
    # - dose-response-absic50-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png
    util.plot_barplot_replicate_data(
        abs_1, abs_2, abs_3, fig_type="absic50",
        fig_name="-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper",
        **_paper_common,
    )


def generate_dose_response_figures(cfg: DoseResponseConfig) -> None:
    generate_residuals_figures(cfg)
    generate_ic50_dmax_r2_figures(cfg)

# -----------------------------------------------------------------------
# Stage 3: Curves
# -----------------------------------------------------------------------

def _write_curve_example_metadata(
    cfg: DoseResponseConfig,
    *,
    params,
    df_params,
    curve_records,
    error_type,
    scenario_key,
    compounds,
    concentrations,
    replicates,
    dilution,
    expected_noise,
    min_dist,
) -> None:
    """Write fit metadata consumed by the tables-stage curve wrapper."""
    import json

    metadata_path = (
        cfg.paper_figures_dir
        / "dose_response_curve_examples.json"
    )

    records = []

    for record in curve_records:
        fit_table = record["fit_table"]

        for compound, row in fit_table.iterrows():
            true = df_params.loc[compound]

            obtained_abs_ic50 = dr.IC50(
                row["b"],
                row["c"],
                row["d"],
                row["e"],
            )

            records.append(
                {
                    "layout_key": record["layout_key"],
                    "display_type": record["display_type"],
                    "layout_file": record["layout_file"],
                    "curve_figure_stem": (
                        record["curve_figure_stem"]
                    ),
                    "compound": int(compound),
                    "lost_rows": record["lost_rows"],
                    "true_b": float(true["b"]),
                    "true_c": float(true["c"]),
                    "true_d": float(true["d"]),
                    "true_relative_ic50": float(true["e"]),
                    "true_absolute_ic50": float(
                        true["abs IC50"]
                    ),
                    "fit_b": float(row["b"]),
                    "fit_c": float(row["c"]),
                    "fit_d": float(row["d"]),
                    "fit_relative_ic50": float(row["e"]),
                    "fit_absolute_ic50": (
                        None
                        if not np.isfinite(obtained_abs_ic50)
                        else float(obtained_abs_ic50)
                    ),
                    "r2_score": (
                        None
                        if not np.isfinite(row["r2_score"])
                        else float(row["r2_score"])
                    ),
                }
            )

    metadata = {
        "scenario_key": scenario_key,
        "disturbance_type": error_type["type"],
        "disturbance_strength": error_type["error"],
        "negative_controls_affected": (
            error_type["neg_controls_affected"]
        ),
        "plate_rows": 16,
        "plate_columns": 24,
        "negative_controls": 20,
        "compounds": compounds,
        "concentrations": concentrations,
        "replicates": replicates,
        "dilution": dilution,
        "start_dose": 10000.0,
        "measurement_noise_amplitude": expected_noise,
        "min_dist": min_dist,
        "records": records,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Written:", metadata_path)


def generate_example_curves(cfg: DoseResponseConfig) -> None:
    """Generate matched illustrative DR curve panels for all layout families.

    All panels use the same simulated compounds, dose-response parameters,
    plate content, disturbance, and normalization procedure. Layout geometry
    is therefore the only layout-dependent input.
    """
    np.random.seed(42)

    cfg.paper_figures_dir.mkdir(parents=True, exist_ok=True)

    compounds = 12
    concentrations = 8
    replicates = 3
    dilution = dilution_for(concentrations)

    slopes = [0.5, 1.0, 1.5]
    current_e = 50.0
    expected_noise = 0.01
    min_dist = 0

    error_type = {
        "type": "right-half",
        "error_function": dt.add_errors_to_right_columns_half,
        "error": 0.4,
        "neg_controls_affected": True,
    }

    scenario_key = CURVE_FIGURE_SCENARIO_KEY
    limits = [{"from": 15, "to": 16}]

    params = [
        {
            "compound": compound,
            "b": slopes[compound % len(slopes)],
            "c": 0.0,
            "d": 100.0,
            "e": current_e + 5.0 * np.random.random(),
            "startDose": 10000.0,
            "nDose": concentrations,
            "dilution": dilution,
        }
        for compound in range(compounds)
    ]

    df_params = pd.DataFrame.from_dict(params).set_index("compound")
    df_params["abs IC50"] = dr.IC50(
        df_params["b"],
        df_params["c"],
        df_params["d"],
        df_params["e"],
    )

    plate_content = dr.generate_plate_content(
        dose_response_params=params,
        replicates=replicates,
    )

    selected_specs = [
        spec
        for spec in sorted(
            DOSE_RESPONSE_LAYOUT_SPECS,
            key=lambda spec: spec.plot_order,
        )
        if spec.curve_example_file is not None
    ]

    curve_records = []

    for layout_spec in selected_specs:
        layout_type = layout_spec.display_type
        layout_dir = layout_spec.layout_dir
        layout_file = layout_spec.curve_example_file

        print("Generating illustrative curves:", layout_type)

        normalization_function = (
            layout_spec._resolved_error_correction()
        )

        figure_stem = dr_curve_stem(
            layout_spec.key,
            layout_file,
            scenario_key,
        )

        plate_type_dict = {
            "type": layout_type,
            "dir": layout_dir,
            "error_correction": normalization_function,
            "requires_layout_update": (
                layout_spec.requires_layout_update
            ),
        }

        for limit in limits:
            fit_table = dr.plate_curves_after_error(
                layout_dir,
                layout_file,
                plate_content.copy(),
                expected_noise,
                error_type["error_function"],
                error_type["error"],
                normalization_function,
                min_dist,
                lose_from_row=limit["from"],
                lose_to_row=limit["to"],
                neg_controls_affected=(
                    error_type["neg_controls_affected"]
                ),
                df_params=df_params,
                plate_type=plate_type_dict,
                compounds=compounds,
                concentrations=concentrations,
                replicates=replicates,
                curve_figure_stem=figure_stem,
                output_dir=str(cfg.paper_figures_dir),
            )

            curve_records.append(
                {
                    "layout_key": layout_spec.key,
                    "display_type": layout_spec.display_type,
                    "layout_file": layout_file,
                    "curve_figure_stem": figure_stem,
                    "lost_rows": limit["to"] - limit["from"],
                    "fit_table": fit_table,
                }
            )

    _write_curve_example_metadata(
        cfg,
        params=params,
        df_params=df_params,
        curve_records=curve_records,
        error_type=error_type,
        scenario_key=scenario_key,
        compounds=compounds,
        concentrations=concentrations,
        replicates=replicates,
        dilution=dilution,
        expected_noise=expected_noise,
        min_dist=min_dist,
    )

# -----------------------------------------------------------------------
# Stage 4: Tables
# -----------------------------------------------------------------------


def generate_ic50_latex_tables(cfg: DoseResponseConfig) -> None:
    """
    Generate ic50-pvalues-half-columns-neg-controls-0.4.tex
    (b_table_stats Table 1) from the 8-dose, error=0.4,
    right-half-neg-control scenario CSVs.

    Table construction (Welch t-test, \\multirow formatting, scientific
    notation) lives in util.write_latex_ic50_pvalue_table so it is shared
    with any future benchmark that needs the same output structure.
    """
    cfg.latex_tables_dir.mkdir(parents=True, exist_ok=True)

    doses, dilution, error_nl = 8, 8, 0.4
    id_text = "right-half-neg-control-log-new-reg"

    rel_1, rel_2, rel_3 = _load_csv_triple(
        cfg, "relative_ic50_data", doses, dilution, error_nl, id_text
    )
    abs_1, abs_2, abs_3 = _load_csv_triple(
        cfg, "absolute_ic50_data", doses, dilution, error_nl, id_text
    )

    out_path = cfg.latex_tables_dir / "ic50-pvalues-half-columns-neg-controls-0.4.tex"
    util.write_latex_ic50_pvalue_table(
        rel_1, rel_2, rel_3,
        abs_1, abs_2, abs_3,
        path=out_path,
    )



def _write_dr_table_pair(
    data_1rep: "np.ndarray",
    data_2rep: "np.ndarray",
    data_3rep: "np.ndarray",
    stem: str,
    latex_tables_dir: "Path",
    prefix: str = "ic50",
) -> None:
    """Write mean-std and p-value table fragments for one DR scenario + metric.

    Filenames written:
      ``{stem}-mean-std.tex``
      ``{stem}-pvalues.tex``

    Parameters
    ----------
    prefix :
        ``"residuals"`` or ``"ic50"`` (covers both relative and absolute IC50).
        Determines which stacker and value column are used:
        - ``"residuals"``: uses ``_stack_replicate_residuals_frames``,
          value column ``"true_residuals"``
        - ``"ic50"``: uses ``_stack_replicate_results_frames``,
          value column ``"MSE"``
    """
    if prefix == "residuals":
        df = util._stack_replicate_residuals_frames([data_1rep, data_2rep, data_3rep])
        value_col = "true_residuals"
    else:
        df = util._stack_replicate_results_frames([data_1rep, data_2rep, data_3rep])
        value_col = "MSE"

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    layouts = DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER
    pairs = [
        (layouts[i], layouts[j])
        for i in range(len(layouts))
        for j in range(i + 1, len(layouts))
    ]

    util.write_latex_mean_std_table(
        df=df,
        group_col="replicates",
        value_col=value_col,
        group_values=[1, 2, 3],
        layouts=layouts,
        path=latex_tables_dir / f"{stem}-mean-std.tex",
        group_label="Replicates",
        bold_min=True,
    )
    util.write_latex_pvalue_table(
        df=df,
        group_col="replicates",
        value_col=value_col,
        group_values=[1, 2, 3],
        layout_pairs=pairs,
        path=latex_tables_dir / f"{stem}-pvalues.tex",
        group_label="Comparison",
    )


def generate_dr_full_latex_tables(cfg: "DoseResponseConfig") -> None:
    """Generate mean-std and p-value LaTeX table fragments for all
    dose-response scenarios (residuals, relative IC50, absolute IC50).

    For each combination of scenario group × (doses, dilution) × error level
    six files are written to cfg.latex_tables_dir:

      dr-residuals-mean-std-{doses}doses-dil{dil}-{label}-{error}.tex
      dr-residuals-pvalues-{doses}doses-dil{dil}-{label}-{error}.tex
      dr-relic50-mean-std-{doses}doses-dil{dil}-{label}-{error}.tex
      dr-relic50-pvalues-{doses}doses-dil{dil}-{label}-{error}.tex
      dr-absic50-mean-std-{doses}doses-dil{dil}-{label}-{error}.tex
      dr-absic50-pvalues-{doses}doses-dil{dil}-{label}-{error}.tex
    """
    cfg.latex_tables_dir.mkdir(parents=True, exist_ok=True)
    d = cfg.latex_tables_dir

    # Map scenario id_text → short label for filenames (must match b_table_stats.tex)
    LABEL_MAP = DR_STEM_LABEL_BY_ID

    for id_text, error_nls, _fig_fn, _r2_fn in IC50_DMAX_R2_SCENARIO_GROUPS:
        label = LABEL_MAP[id_text]
        for doses, dilution in DOSE_RESPONSE_FIGURE_CASES:
            for error_nl in error_nls:
                stem_base = f"{doses}doses-dil{dilution}-{label}-{error_nl}"

                r1, r2, r3 = _load_csv_triple(
                    cfg, "residuals", doses, dilution, error_nl, id_text
                )
                _write_dr_table_pair(r1, r2, r3,
                                     stem=f"dr-residuals-{stem_base}",
                                     latex_tables_dir=d,
                                     prefix="residuals")

                rel1, rel2, rel3 = _load_csv_triple(
                    cfg, "relative_ic50_data", doses, dilution, error_nl, id_text
                )
                _write_dr_table_pair(rel1, rel2, rel3,
                                     stem=f"dr-relic50-{stem_base}",
                                     latex_tables_dir=d,
                                     prefix="ic50")

                abs1, abs2, abs3 = _load_csv_triple(
                    cfg, "absolute_ic50_data", doses, dilution, error_nl, id_text
                )
                _write_dr_table_pair(abs1, abs2, abs3,
                                     stem=f"dr-absic50-{stem_base}",
                                     latex_tables_dir=d,
                                     prefix="ic50")


def _build_dr_overview_df_for_scenario(
    cfg: "DoseResponseConfig",
    prefix: str,
    id_text: str,
    error_nls,
) -> "pd.DataFrame":
    """Collect dose-response data for ONE scenario into a long-form DataFrame.

    Parameters
    ----------
    prefix : {"residuals", "relic50", "absic50"}
    id_text : scenario key, e.g. "curve_info-new-reg"
    error_nls : iterable of error levels for this scenario

    Returns
    -------
    DataFrame with columns:
      doses, dilution, error_nl, replicates, layout, value
    """
    data_prefix_map = {
        "dmax":      "absolute_ic50_data",
        "residuals": "residuals",
        "relic50":   "relative_ic50_data",
        "absic50":   "absolute_ic50_data",
    }
    value_col_map = {
        "dmax":      "diff_d",
        "residuals": "true_residuals",
        "relic50":   "MSE",
        "absic50":   "MSE",
    }
    
    csv_prefix    = data_prefix_map[prefix]
    value_col     = value_col_map[prefix]
    scenario_label = DR_LABEL_BY_ID[id_text]
    layouts = DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER

    rows = []
    for doses, dilution in DOSE_RESPONSE_FIGURE_CASES:
        for error_nl in error_nls:
            try:
                d1, d2, d3 = _load_csv_triple(
                    cfg, csv_prefix, doses, dilution, error_nl, id_text
                )
            except Exception as exc:
                print(
                    f"  WARNING: skipping {scenario_label} "
                    f"{doses}d dil{dilution} e{error_nl}: {exc}"
                )
                continue

            if prefix == "residuals":
                df_long = util._stack_replicate_residuals_frames([d1, d2, d3])
            else:
                df_long = util._stack_replicate_results_frames([d1, d2, d3])
            if prefix == "dmax":
                for col in ("d", "fit_d"):
                    df_long[col] = pd.to_numeric(df_long[col], errors="coerce")
                df_long["diff_d"] = (df_long["d"] - df_long["fit_d"]).abs()

            df_long[value_col] = pd.to_numeric(df_long[value_col], errors="coerce")
            df_long = df_long.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[value_col]
            )

            for lay in layouts:
                sub = df_long[df_long["layout"] == lay]
                for rep in [1, 2, 3]:
                    vals = sub.loc[sub["replicates"] == rep, value_col]
                    rows.append(
                        dict(
                            doses=doses,
                            dilution=dilution,
                            error_nl=error_nl,
                            replicates=rep,
                            layout=lay,
                            value=float(vals.mean()) if not vals.empty else float("nan"),
                        )
                    )
    return pd.DataFrame(rows)


def _build_dr_overview_df(
    cfg: "DoseResponseConfig",
    prefix: str,
) -> "pd.DataFrame":
    """Collect data for ALL scenarios (kept for backward compatibility).

    Calls _build_dr_overview_df_for_scenario for each scenario group and
    adds a 'scenario_label' column.  Existing code that calls this function
    directly is unaffected.
    """
    frames = []
    for id_text, error_nls, _fig_fn, _r2_fn in IC50_DMAX_R2_SCENARIO_GROUPS:
        df = _build_dr_overview_df_for_scenario(cfg, prefix, id_text, error_nls)
        if not df.empty:
            df.insert(0, "scenario_label", DR_LABEL_BY_ID[id_text])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_dr_overview_table(
    df: "pd.DataFrame",
    layouts: "list[str]",
    path: "Path",
    bold_min: bool = True,
    show_scenario_col: bool = True,
) -> None:
    """Write a compact overview LaTeX tabular fragment from *df*.

    Parameters
    ----------
    show_scenario_col : bool
        True  (default) – include a leading "Scenario" column and group rows
                          by scenario_label.  Use when df covers multiple
                          scenarios (backward-compatible behaviour).
        False           – omit the Scenario column entirely.  Use when each
                          file covers exactly one scenario; the scenario name
                          belongs in the LaTeX caption instead.

    Row structure when show_scenario_col=False:
      Top-level groups: (doses, dilution, error_nl).  A \\cmidrule separates
      groups.  Within each group: one row per replicate count (1, 2, 3).

    Row structure when show_scenario_col=True (original behaviour):
      Same as above, but wrapped in an outer scenario grouping separated by
      \\midrule.

    Columns: [Scenario |] Doses | Dil.\\ (1:N) | Error | Rep. | <layouts>

    Best layout per row is bolded.  COMPD gets a significance superscript
    (* / ** / ***) vs PLAID from a Welch t-test.
    """
    from scipy import stats as _st

    def _sig_flag(a, b):
        if len(a) < 2 or len(b) < 2:
            return ""
        _, p = _st.ttest_ind(a, b, equal_var=False)
        if p < 0.001: return r"$^{***}$"
        if p < 0.01:  return r"$^{**}$"
        if p < 0.05:  return r"$^{*}$"
        return ""

    replicates_list = sorted(df["replicates"].unique())

    if show_scenario_col:
        n_data_cols = 5 + len(layouts)
        col_spec    = "lllcc" + "c" * len(layouts)
        cmidrule    = rf"\cmidrule{{2-{n_data_cols}}}"
        header      = (
            r"Scenario & Doses & Dil.\ (1:N) & Error & Rep. & "
            + " & ".join(layouts) + r" \\"
        )
    else:
        n_data_cols = 4 + len(layouts)
        col_spec    = "lccc" + "c" * len(layouts)
        cmidrule    = rf"\cmidrule{{1-{n_data_cols}}}"
        header      = (
            r"Doses & Dil.\ (1:N) & Error & Rep. & "
            + " & ".join(layouts) + r" \\"
        )

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    def _emit_sub_groups(sub_df, prefix_col_fn):
        """Emit rows for one block of (doses, dilution, error_nl) sub-groups."""
        seen = {}
        for _, row in sub_df.iterrows():
            key = (row["doses"], row["dilution"], row["error_nl"])
            seen[key] = None
        sub_groups = list(seen.keys())

        for sg_idx, (doses, dil, enl) in enumerate(sub_groups):
            if sg_idx > 0:
                lines.append(cmidrule)

            for ri, rep in enumerate(replicates_list):
                prefix_cells = prefix_col_fn(sg_idx, ri, doses, dil, enl)
                rep_cell     = str(int(rep))

                sub = sub_df[
                    (sub_df["doses"]      == doses)
                    & (sub_df["dilution"] == dil)
                    & (sub_df["error_nl"] == enl)
                    & (sub_df["replicates"] == rep)
                ]
                lay_vals  = {
                    lay: sub[sub["layout"] == lay]["value"].dropna().tolist()
                    for lay in layouts
                }
                lay_means = {
                    lay: float(np.nanmean(v)) if v else float("nan")
                    for lay, v in lay_vals.items()
                }

                valid = [m for m in lay_means.values() if not np.isnan(m)]
                best  = (min(valid) if bold_min else max(valid)) if valid else float("nan")

                row_cells = list(prefix_cells) + [rep_cell]
                for lay in layouts:
                    m = lay_means[lay]
                    if np.isnan(m):
                        cell = "--"
                    else:
                        val_str = f"{m:.4f}"
                        cell    = rf"\textbf{{{val_str}}}" if m == best else val_str
                        if lay == "COMPD":
                            cell += _sig_flag(
                                lay_vals.get("PLAID", []),
                                lay_vals.get("COMPD", []),
                            )
                    row_cells.append(cell)

                lines.append(" & ".join(row_cells) + r" \\")

    if show_scenario_col:
        scenarios = list(dict.fromkeys(df["scenario_label"]))
        for s_idx, scenario in enumerate(scenarios):
            if s_idx > 0:
                lines.append(r"\midrule")
            sc_df = df[df["scenario_label"] == scenario]

            def prefix_col_fn(sg_idx, ri, doses, dil, enl,
                              _s=scenario):  # capture scenario
                return [
                    _s if (sg_idx == 0 and ri == 0) else "",
                    str(int(doses)) if ri == 0 else "",
                    str(int(dil))   if ri == 0 else "",
                    str(enl)        if ri == 0 else "",
                ]

            _emit_sub_groups(sc_df, prefix_col_fn)
    else:
        def prefix_col_fn(sg_idx, ri, doses, dil, enl):
            return [
                str(int(doses)) if ri == 0 else "",
                str(int(dil))   if ri == 0 else "",
                str(enl)        if ri == 0 else "",
            ]

        _emit_sub_groups(df, prefix_col_fn)

    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  Written: {path}")


def generate_dr_overview_tables(cfg: "DoseResponseConfig") -> None:
    """Generate compact DR overview tables — one file per scenario per metric.

    Nine files are written to cfg.latex_tables_dir:

      dr-overview-dmax-bowl.tex
      dr-overview-dmax-bowl-neg.tex
      dr-overview-dmax-column.tex
      dr-overview-residuals-bowl.tex       (bowl, neg unaffected)
      dr-overview-residuals-bowl-neg.tex   (bowl, neg affected)
      dr-overview-residuals-column.tex     (half-plate column)
      dr-overview-rel-ic50-bowl.tex
      dr-overview-rel-ic50-bowl-neg.tex
      dr-overview-rel-ic50-column.tex
      dr-overview-abs-ic50-bowl.tex
      dr-overview-abs-ic50-bowl-neg.tex
      dr-overview-abs-ic50-column.tex

    Each file contains one LaTeX tabular sized to fit on a single page:
    rows are (doses x dilution x error_nl x replicate), columns are layouts.
    The scenario name is not a column — it belongs in the LaTeX caption.

    The old three-file output (dr-overview-*.tex without scenario suffix) is
    no longer written; update \\input{} calls in 0_supplement.tex accordingly.
    """
    layouts = DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER
    d = cfg.latex_tables_dir
    d.mkdir(parents=True, exist_ok=True)

    for prefix, metric_stem, bold_min in [
        ("dmax",      "dr-overview-dmax",      True),
        ("residuals", "dr-overview-residuals", True),
        ("relic50",   "dr-overview-rel-ic50",  True),
        ("absic50",   "dr-overview-abs-ic50",  True),
    ]:
        for id_text, error_nls, _fig_fn, _r2_fn in IC50_DMAX_R2_SCENARIO_GROUPS:
            suffix = DR_FILE_SUFFIX_BY_ID[id_text]
            fname  = f"{metric_stem}-{suffix}.tex"
            print(f"\nBuilding {fname} ...")

            df = _build_dr_overview_df_for_scenario(cfg, prefix, id_text, error_nls)
            if df.empty:
                print(f"  WARNING: no data for {prefix}/{id_text} — skipping")
                continue

            _write_dr_overview_table(
                df,
                layouts,
                path=d / fname,
                bold_min=bold_min,
                show_scenario_col=False,
            )


def generate_curve_figure_tex(cfg, metadata_path=None):
    """Write compound-wise curve comparisons from existing JSON metadata and PNGs.

    Replace the existing writer with this function. The tables-stage call and
    dr_section_auto.tex inclusion remain unchanged. No simulation or fitting
    functions are called.
    """
    import json
    import math
    import re
    from pathlib import Path
    from benchmark_common import DOSE_RESPONSE_LAYOUT_SPECS, dr_curve_png_name

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    def number(value, digits=2):
        value = finite(value)
        return rf"{value:.{digits}f}" if value is not None else r"\mathrm{undefined}"

    def short(value):
        value = finite(value)
        return f"{value:g}" if value is not None else 'unspecified'

    def tex(value):
        replacements = {
            '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
            '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
            '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        }
        return ''.join(replacements.get(character, character) for character in str(value))

    def slug(value):
        return re.sub(r'[^a-z0-9]+', '-', str(value).lower()).strip('-')

    header_fields = {
        'scenario_key', 'records', 'compounds', 'concentrations', 'replicates',
        'plate_rows', 'plate_columns', 'negative_controls',
        'disturbance_type', 'disturbance_strength',
    }
    record_fields = {
        'layout_key', 'display_type', 'curve_figure_stem', 'compound',
        'true_relative_ic50', 'true_absolute_ic50',
        'fit_relative_ic50', 'fit_absolute_ic50', 'r2_score',
    }

    def is_curve_metadata(value):
        return (
            isinstance(value, dict) and header_fields.issubset(value)
            and isinstance(value['records'], list) and bool(value['records'])
            and isinstance(value['records'][0], dict)
            and record_fields.issubset(value['records'][0])
        )

    def load_metadata(path):
        with path.open(encoding='utf-8-sig') as handle:
            value = json.load(handle)
        require(is_curve_metadata(value), f'Not a curve metadata file: {path}')
        return value

    figure_dir = Path(cfg.paper_figures_dir).resolve()
    if metadata_path is not None:
        source_path = Path(metadata_path).resolve()
        metadata = load_metadata(source_path)
    else:
        # Search existing output locations; do not guess a metadata filename.
        roots = {figure_dir, figure_dir.parent}
        if hasattr(cfg, 'data_dir'):
            data_dir = Path(cfg.data_dir).resolve()
            roots.update((data_dir, data_dir.parent))
        candidates = set(Path.cwd().glob('*.json'))
        for root in roots:
            if root.is_dir():
                candidates.update(root.rglob('*.json'))
        matches = []
        parse_errors = []
        for path in sorted({path.resolve() for path in candidates}):
            try:
                with path.open(encoding='utf-8-sig') as handle:
                    value = json.load(handle)
            except (OSError, ValueError) as exc:
                parse_errors.append(f'{path}: {exc}')
                continue
            if is_curve_metadata(value):
                matches.append((path, value))
        require(bool(matches),
                'No curve-metadata JSON found in the configured output locations. '
                'Run --stage curves first, or pass its existing JSON path explicitly '
                'as generate_curve_figure_tex(cfg, metadata_path=...). '
                + ('JSON parse errors: ' + '; '.join(parse_errors[:3]) if parse_errors else ''))
        require(len(matches) == 1,
                'Multiple curve-metadata JSON files match; refusing to choose a stale '
                'or different experiment. Pass the intended existing path explicitly: '
                + ', '.join(str(path) for path, _ in matches))
        source_path, metadata = matches[0]

    records = metadata['records']
    by_panel = {}
    for record in records:
        require(isinstance(record, dict) and record_fields.issubset(record),
                'A curve record is missing required identity or statistics fields.')
        compound = int(record['compound'])
        require(compound == record['compound'] and compound >= 0,
                f'Invalid compound index: {record["compound"]!r}')
        identity = (str(record['layout_key']), compound)
        require(identity not in by_panel,
                f'Duplicate layout/compound record {identity}; do not mix row-loss '
                'variants or multiple experiments in one metadata file.')
        by_panel[identity] = record

    specs = sorted(
        (spec for spec in DOSE_RESPONSE_LAYOUT_SPECS
         if spec.curve_example_file is not None),
        key=lambda spec: spec.plot_order,
    )
    require(bool(specs), 'No illustrative layout families are configured.')
    layout_keys = [str(spec.key) for spec in specs]
    require(len(layout_keys) == len(set(layout_keys)), 'Duplicate registry layout keys.')
    available_keys = {key for key, _ in by_panel}
    require(set(layout_keys).issubset(available_keys),
            'Curve metadata is missing configured layout families: '
            + ', '.join(sorted(set(layout_keys) - available_keys)))

    count = int(metadata['compounds'])
    require(count > 0, 'Metadata compounds must be positive.')
    require(all(compound < count for _, compound in by_panel),
            'A compound index exceeds the metadata compound count.')
    selection = globals().get('CURVE_FIGURE_COMPOUNDS', None)
    compounds = list(range(count)) if selection is None else list(selection)
    require(bool(compounds), 'No compounds selected for the curve report.')
    require(all(isinstance(compound, int) and not isinstance(compound, bool)
                and 0 <= compound < count for compound in compounds),
            'CURVE_FIGURE_COMPOUNDS must be None or a list of valid integer indices.')
    require(len(compounds) == len(set(compounds)), 'Duplicate selected compound indices.')
    missing_records = [(key, compound) for compound in compounds for key in layout_keys
                       if (key, compound) not in by_panel]
    require(not missing_records,
            f'Missing curve records for selected panels: {missing_records}. '
            'The producer must retain records for unavailable fits too.')

    base = (
        f'Illustrative dose-response curves for synthetic compound index {{compound}} '
        f'on {int(metadata["plate_rows"])}' + r'$\times$'
        + f'{int(metadata["plate_columns"])}-well plates containing '
        f'{count} compounds, {int(metadata["concentrations"])} concentrations per compound, '
        f'{int(metadata["replicates"])} technical replicates per concentration, and '
        f'{int(metadata["negative_controls"])} physical negative-control wells. '
        f'Disturbance: {tex(metadata["disturbance_type"])} '
        f'(strength parameter $s={short(metadata["disturbance_strength"])}$). '
    )
    if 'negative_controls_affected' in metadata:
        base += ('Negative controls are subject to disturbance application. '
                 if metadata['negative_controls_affected']
                 else 'Pre-disturbance negative-control signals are retained before normalization. ')
    if 'start_dose' in metadata:
        base += f'Starting concentration: {short(metadata["start_dose"])}'
        if metadata.get('dose_unit'):
            base += ' ' + tex(metadata['dose_unit'])
        base += '. '
    if 'dilution' in metadata:
        base += f'Dilution factor: {short(metadata["dilution"])}. '
    if 'measurement_noise_amplitude' in metadata:
        base += ('Configured measurement-noise parameter: '
                 f'{short(metadata["measurement_noise_amplitude"])}. ')
    normalization = metadata.get('normalization_name', metadata.get('normalization'))
    if normalization:
        base += 'Normalization: ' + tex(normalization) + '. '
    if 'control_pseudo_observations' in metadata:
        base += f'The fit includes {int(metadata["control_pseudo_observations"])} control pseudo-observations'
        if 'controls_per_pseudo_observation' in metadata:
            base += f', each averaging {int(metadata["controls_per_pseudo_observation"])} normalized controls'
        base += '. '

    lines = [
        '% AUTO-GENERATED by run_dose_response_benchmark.py --stage tables.',
        '% Statistics and experiment settings are read from existing curve metadata.',
        '% This writer does not simulate plates, refit curves, or regenerate PNGs.',
        '',
    ]
    scenario_label = slug(metadata['scenario_key'])
    used_files = set()
    warning_count = 0
    for figure_index, compound in enumerate(compounds):
        panel_records = [by_panel[(key, compound)] for key in layout_keys]
        lines += [r'\begin{figure*}[!htbp]', r'  \centering']
        for panel_index, (key, record) in enumerate(zip(layout_keys, panel_records)):
            filename = dr_curve_png_name(record['curve_figure_stem'], compound)
            require(filename not in used_files, f'Curve PNG filename collision: {filename}')
            used_files.add(filename)
            path = figure_dir / filename
            lines += [r'  \begin{subfigure}[t]{0.48\textwidth}', r'    \centering']
            if path.is_file():
                lines.append(rf'    \includegraphics[width=\linewidth]{{figures/{filename}}}')
            else:
                fit_fields = ('fit_b', 'fit_c', 'fit_d', 'fit_relative_ic50')
                fit_unavailable = (all(field in record for field in fit_fields)
                                   and any(finite(record[field]) is None for field in fit_fields))
                if not fit_unavailable:
                    raise FileNotFoundError(
                        f'Metadata contains fitted results but the corresponding PNG '
                        f'is missing: {path}. Regenerate --stage curves; no benchmark '
                        'simulation is invoked by this writer.')
                warning_count += 1
                print(f'WARNING: no available fitted curve for {key}, compound {compound}')
                lines += [
                    r'    \fbox{\parbox[c][35mm][c]{0.9\linewidth}{\centering Curve unavailable}}',
                    f'    % MISSING FITTED CURVE: {filename}',
                ]

            true_relative = finite(record['true_relative_ic50'])
            true_absolute = finite(record['true_absolute_ic50'])
            if (true_relative is not None and true_absolute is not None
                    and math.isclose(true_relative, true_absolute, rel_tol=1e-12, abs_tol=1e-12)):
                truth = rf'$\mathrm{{IC}}_{{50,\mathrm{{true}}}}={number(true_relative)}$'
            else:
                truth = (rf'$\mathrm{{IC}}_{{50,\mathrm{{rel,true}}}}={number(true_relative)}$, '
                         rf'$\mathrm{{IC}}_{{50,\mathrm{{abs,true}}}}={number(true_absolute)}$')
            caption = (
                tex(record['display_type']) + ': '
                + rf'$R^2={number(record["r2_score"], 4)}$. '
                + truth + '. '
                + rf'$\widehat{{\mathrm{{IC}}}}_{{50,\mathrm{{rel}}}}={number(record["fit_relative_ic50"])}$, '
                + rf'$\widehat{{\mathrm{{IC}}}}_{{50,\mathrm{{abs}}}}={number(record["fit_absolute_ic50"])}$.'
            )
            lines += [rf'    \caption{{{caption}}}',
                      rf'    \label{{fig:dr-curve-{scenario_label}-compound-{compound}-{slug(key)}}}',
                      r'  \end{subfigure}']
            if panel_index + 1 < len(panel_records):
                lines.append(r'  \hfill' if panel_index % 2 == 0 else r'  \par\medskip')

        experiment_caption = base.replace('{compound}', str(compound))
        losses = {record.get('lost_rows') for record in panel_records}
        if len(losses) == 1 and None not in losses:
            experiment_caption += f'Row-loss setting: {short(next(iter(losses)))} array row(s). '
        if 'lost_active_rows' in metadata:
            experiment_caption += f'Active rows removed: {int(metadata["lost_active_rows"])}. '
        experiment_caption += (
            'Panels show the simulated observations, original and fitted LL.4 curves, '
            'and control pseudo-observations. Panel captions report fitted $R^2$ and '
            'true and estimated relative/absolute potency; undefined values are marked explicitly. '
        )
        truth_fields = ('true_b', 'true_c', 'true_d', 'true_relative_ic50', 'true_absolute_ic50')
        truth_vectors = [[finite(record.get(field)) for field in truth_fields]
                         for record in panel_records]
        if all(all(value is not None for value in vector) for vector in truth_vectors):
            shared_truth = all(all(math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
                                  for value, reference in zip(vector, truth_vectors[0]))
                               for vector in truth_vectors[1:])
            experiment_caption += (
                'The panels share the same ground-truth curve parameters. '
                if shared_truth else
                'The panels have different ground-truth curve parameters; true potency '
                'is therefore reported individually rather than as one figure-wide value. '
            )
        lines += [rf'  \caption{{{experiment_caption}}}',
                  rf'  \label{{fig:dr-curves-{scenario_label}-compound-{compound}}}']
        if figure_index == 0:
            lines.append(r'  \label{fig:dose-response-illustrative-curves}')
        lines += [r'\end{figure*}', '', r'\clearpage', '']

    tex_name = globals().get('CURVE_FIGURE_TEX_NAME', 'dose_response_illustrative_curves.tex')
    output_path = figure_dir.parent / 'tikz-figures' / tex_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print('Curve metadata:', source_path)
    print(f'Written: {output_path} ({len(compounds)} figures, '
          f'{len(used_files)} panels, {warning_count} unavailable fitted curves)')
    return output_path


def generate_dr_section_tex(cfg: "DoseResponseConfig") -> None:
    """Write tikz-figures/dr_section_auto.tex.

    AUTO-GENERATED — do not edit by hand.
    Re-run with:  python run_dose_response_benchmark.py --stage tables

    Structure
    ---------
    \\subsection{<long_label> plate effects}
      \\subsubsection{d_max errors}
        figure* (grid: doses × error levels)
      \\input{tables/dr-overview-rel-ic50-<suffix>}
      \\subsubsection{Relative IC50 errors}
        figure*
      \\input{tables/dr-overview-abs-ic50-<suffix>}
      \\subsubsection{Absolute IC50 errors}
        figure*
      \\subsubsection{Residuals}
        figure*
      \\input{tables/dr-overview-residuals-<suffix>}
      \\subsubsection{Fraction of poor fits}   [bowl only]
        figure*
    """
    tikz_dir = Path("detailed-experimental-results-source") / "tikz-figures"
    tikz_dir.mkdir(parents=True, exist_ok=True)
    out_path = tikz_dir / "dr_section_auto.tex"

    results_root = Path("detailed-experimental-results-source")
    fig_root     = results_root / "figures"
    tbl_root     = results_root / "tables"

    def _tbl_line(stem: str, disturbance_desc: str) -> str:
        full = tbl_root / f"{stem}.tex"
        if not full.exists():
            return f"% MISSING: tables/{stem}.tex"
        # stem is e.g. "dr-overview-rel-ic50-bowl-neg"
        # metric key is the part between "dr-overview-" and the suffix
        for key in ("dmax", "rel-ic50", "abs-ic50", "residuals"):
            if f"-{key}-" in stem:
                template = _DR_TABLE_CAPTIONS[key]
                break
        else:
            template = "Overview table. " + _DR_CAPTION_PLACEHOLDER + "."
        caption = template.replace(_DR_CAPTION_PLACEHOLDER, disturbance_desc)
        label   = "tab:" + stem
        return "\n".join([
            r"\begin{table}[H]",
            r"  \caption{%",
            f"    {caption}",
            r"  }",
            rf"  \label{{{label}}}",
            r"  \centering",
            rf"  \input{{tables/{stem}}}",
            r"\end{table}",
        ])

    # Lookup dicts built from the module-level lambda groups
    _ic50_fn_by_id = {g[0]: g[2] for g in IC50_DMAX_R2_SCENARIO_GROUPS}
    _r2_fn_by_id   = {g[0]: g[3] for g in IC50_DMAX_R2_SCENARIO_GROUPS}
    _res_fn_by_id  = {g[0]: g[2] for g in RESIDUALS_SCENARIO_GROUPS}

    lines: list[str] = [
        "% AUTO-GENERATED by run_dose_response_benchmark.py -- DO NOT EDIT.",
        "% Regenerate with:  python run_dose_response_benchmark.py --stage tables",
        "%",
    ]

    for idx, d in enumerate(dr_scenarios()):
        id_text      = d.dr_id_text
        emph         = d.emph_name
        label        = d.long_label
        suffix       = d.dr_file_suffix
        ic50_fn      = _ic50_fn_by_id[id_text]
        r2_fn        = _r2_fn_by_id[id_text]
        res_fn       = _res_fn_by_id[id_text]
        is_bowl      = "right-half" not in id_text
        error_levels = d.dr_error_levels

        # --- subsection header ---
        if idx == 0:
            lines += [
                "",
                rf"\subsection{{\emph{{{emph}}} plate effects ({label})}}",
                "",
            ]
        else:
            lines += [
                "",
                r"\clearpage",
                rf"\subsection{{\emph{{{emph}}} plate effects ({label})}}",
                "",
            ]

        # Shared helper: build filenames_by_row + col_labels for any metric
        def _build_rows(png_fn, _error_levels=error_levels):
            """Return (col_labels, filenames_by_row) for given filename function."""
            col_labels = [lv.latex_col_label() for lv in _error_levels]
            rows = []
            for doses, dilution in DOSE_RESPONSE_FIGURE_CASES:
                row_lbl = f"{doses} doses"
                pngs    = [png_fn(doses, dilution, lv.value) for lv in _error_levels]
                rows.append((row_lbl, pngs))
            return col_labels, rows

        def _disturbance_desc():
            return rf"\emph{{{emph}}} plate-effect strengths"

        # ── d_max ──────────────────────────────────────────────────────
        lines += ["", rf"\subsubsection{{{_DR_SUBSUBSECTION['d_diff']}}}", ""]
        col_labels, rows = _build_rows(
            lambda d, dil, enl, _fn=ic50_fn: f"dose-response-d_diff{_fn(d, dil, enl)}.png"
        )
        cap = _DR_CAPTIONS["d_diff"].replace(_DR_CAPTION_PLACEHOLDER, _disturbance_desc())
        lines += util.latex_figure_block(
            fig_root=fig_root, filenames_by_row=rows,
            col_labels=col_labels, caption=cap,
        )
        lines += ["", _tbl_line(f"dr-overview-dmax-{suffix}", _disturbance_desc()), ""]

        # ── rel IC50 ────────────────────────────────────────────────────
        lines += ["", r"\clearpage", rf"\subsubsection{{{_DR_SUBSUBSECTION['relic50']}}}", ""]
        col_labels, rows = _build_rows(
            lambda d, dil, enl, _fn=ic50_fn: f"dose-response-relic50{_fn(d, dil, enl)}.png"
        )
        cap = _DR_CAPTIONS["relic50"].replace(_DR_CAPTION_PLACEHOLDER, _disturbance_desc())
        lines += util.latex_figure_block(
            fig_root=fig_root, filenames_by_row=rows,
            col_labels=col_labels, caption=cap,
        )
        lines += ["", _tbl_line(f"dr-overview-rel-ic50-{suffix}", _disturbance_desc()), ""]

        # ── abs IC50 ────────────────────────────────────────────────────
        lines += ["", r"\clearpage", rf"\subsubsection{{{_DR_SUBSUBSECTION['absic50']}}}", ""]
        col_labels, rows = _build_rows(
            lambda d, dil, enl, _fn=ic50_fn: f"dose-response-absic50{_fn(d, dil, enl)}.png"
        )
        cap = _DR_CAPTIONS["absic50"].replace(_DR_CAPTION_PLACEHOLDER, _disturbance_desc())
        lines += util.latex_figure_block(
            fig_root=fig_root, filenames_by_row=rows,
            col_labels=col_labels, caption=cap,
        )
        lines += ["", _tbl_line(f"dr-overview-abs-ic50-{suffix}", _disturbance_desc()), ""]

        # ── residuals ───────────────────────────────────────────────────
        lines += ["", r"\clearpage", rf"\subsubsection{{{_DR_SUBSUBSECTION['residuals']}}}", ""]
        col_labels, rows = _build_rows(
            lambda d, dil, enl, _fn=res_fn: f"{_fn(d, dil, enl)}.png"
        )
        cap = _DR_CAPTIONS["residuals"].replace(_DR_CAPTION_PLACEHOLDER, _disturbance_desc())
        lines += util.latex_figure_block(
            fig_root=fig_root, filenames_by_row=rows,
            col_labels=col_labels, caption=cap,
        )
        lines += ["", _tbl_line(f"dr-overview-residuals-{suffix}", _disturbance_desc()), ""]

        # ── R^2 (bowl only) ──────────────────────────────────────────────
        if is_bowl:
            lines += ["", r"\clearpage", rf"\subsubsection{{{_DR_SUBSUBSECTION['percentage']}}}", ""]
            col_labels, rows = _build_rows(
                lambda d, dil, enl, _fn=r2_fn: f"percentage-low-r2{_fn(d, dil, enl)}.png"
            )
            cap = _DR_CAPTIONS["percentage"].replace(_DR_CAPTION_PLACEHOLDER, _disturbance_desc())
            lines += util.latex_figure_block(
                fig_root=fig_root, filenames_by_row=rows,
                col_labels=col_labels, caption=cap,
            )
    
    curve_tex_stem = Path(CURVE_FIGURE_TEX_NAME).stem

    lines.extend(
        [
            "",
            r"\clearpage",
            r"\subsection{Illustrative dose-response curves}",
            "",
            rf"\input{{tikz-figures/{curve_tex_stem}}}",
            "",
            r"\clearpage",
            "",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run COMPD/PLAID dose–response benchmark pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["simulate", "figures", "curves", "tables", "all"],
        default="all",
        help=(
            "simulate: generate CSVs; "
            "figures:  generate supplement PNGs; "
            "curves:   generate example curve PNGs; "
            "tables:   generate LaTeX tables; "
            "all:      from simulate to figures to tables to curves"
        ),
    )
    args = parser.parse_args()
    cfg = DoseResponseConfig()

    if args.stage in ("simulate", "all"):
        run_simulations(cfg)
    if args.stage in ("figures", "all"):
        generate_dose_response_figures(cfg)
    if args.stage in ("curves", "all"):
        generate_example_curves(cfg)
    if args.stage in ("tables", "all"):
        generate_ic50_latex_tables(cfg)
        generate_dr_full_latex_tables(cfg)
        generate_dr_overview_tables(cfg)
        generate_curve_figure_tex(cfg)
        generate_dr_section_tex(cfg)


if __name__ == "__main__":
    main()
