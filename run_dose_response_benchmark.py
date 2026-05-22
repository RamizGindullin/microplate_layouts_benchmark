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
  figures  : generate supplement PNGs under generated-plots/dose-response-supplement/
  curves   : generate example curve PNGs under figures/
  all      : simulate → figures → curves  (default)
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
    BOWL_ERROR_LEVELS,
    DOSE_RESPONSE_FIGURE_CASES,
    RIGHT_HALF_ERROR_LEVELS,
    dose_response_curve_examples,
    dose_response_plate_types,
    dilution_for,
    fig_dir_str,
    validate_layout_registry_consistency,
)

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
    """
    id_text: str
    error_nl: float
    error_types: List[Dict[str, Any]]


@dataclass
class DoseResponseConfig:
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

    # Fixed tag matching the 20250706-* filenames in the LaTeX sources.
    date_tag: str = "20250706-"

    scenarios: List[DoseResponseScenario] = field(
        default_factory=lambda: [
            DoseResponseScenario(
                id_text="right-half-neg-control-log-new-reg",
                error_nl=0.2,
                error_types=[{
                    "type": "right-half",
                    "error_function": dt.add_errors_to_right_columns_half,
                    "error_correction": nrm.normalize_plate_nearest_control,
                    "error": 0.2,
                }]
            ),
            # NOTE: scenario with id_text="log-neg-control-new-reg" and
            # label="half-columns-neg-controls-0.4-log" was removed — it was a
            # duplicate of the scenario below (same id_text after correction, same
            # error_nl=0.4, same CSV output), and its label was never used in any
            # figure-generation call.
            DoseResponseScenario(
                id_text="right-half-neg-control-log-new-reg",
                error_nl=0.4,
                error_types=[{
                    "type": "right-half",
                    "error_function": dt.add_errors_to_right_columns_half,
                    "error_correction": nrm.normalize_plate_nearest_control,
                    "error": 0.4,
                }]
            ),
            DoseResponseScenario(
                id_text="curve_info-new-reg",
                error_nl=0.055,
                error_types=[{
                    "type": "bowl-nl",
                    "error_function": dt.add_bowlshaped_errors_nl,
                    "error_correction": nrm.normalize_plate_nearest_control,
                    "error": 0.055,
                }]
            ),
            DoseResponseScenario(
                id_text="bowl-neg-control-new-reg",
                error_nl=0.055,
                error_types=[{
                    "type": "bowl-nl",
                    "error_function": dt.add_bowlshaped_errors_nl,
                    "error_correction": nrm.normalize_plate_nearest_control,
                    "error": 0.055,
                }]
            ),
            DoseResponseScenario(
                id_text="curve_info-new-reg",
                error_nl=0.085,
                error_types=[{
                    "type": "bowl-nl",
                    "error_function": dt.add_bowlshaped_errors_nl,
                    "error_correction": nrm.normalize_plate_nearest_control,
                    "error": 0.085,
                }]
            ),
            DoseResponseScenario(
                id_text="bowl-neg-control-new-reg",
                error_nl=0.085,
                error_types=[{
                    "type": "bowl-nl",
                    "error_function": dt.add_bowlshaped_errors_nl,
                    "error_correction": nrm.normalize_plate_nearest_control,
                    "error": 0.085,
                }]
            ),
        ]
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
        "curve_info-new-reg",
        BOWL_ERROR_LEVELS,
        lambda doses, dil, enl: f"residuals-1-2-3-{doses}doses-dil{dil}-bowl-{enl}",
    ),
    (
        "bowl-neg-control-new-reg",
        BOWL_ERROR_LEVELS,
        lambda doses, dil, enl: f"residuals-1-2-3-{doses}doses-dil{dil}-bowl-neg-controls-{enl}",
    ),
    (
        "right-half-neg-control-log-new-reg",
        RIGHT_HALF_ERROR_LEVELS,
        lambda doses, dil, enl: (
            f"residuals-1-2-3-{doses}doses-dil{dil}-half-columns-neg-controls-{enl}"
        ),
    ),
]


IC50_DMAX_R2_SCENARIO_GROUPS = [
    (
        "curve_info-new-reg",
        BOWL_ERROR_LEVELS,
        lambda doses, dil, enl: f"percentage-low-r2-1-2-3-{doses}doses-dil{dil}-bowl-{enl}",
        lambda doses, dil, enl: f"percentage-low-r2-{doses}doses-dil{dil}-bowl-{enl}",
    ),
    (
        "bowl-neg-control-new-reg",
        BOWL_ERROR_LEVELS,
        lambda doses, dil, enl: f"percentage-low-r2-1-2-3-{doses}doses-dil{dil}-bowl-neg-controls-{enl}",
        lambda doses, dil, enl: f"percentage-low-r2-{doses}doses-dil{dil}-bowl-neg-controls-{enl}",
    ),
    (
        "right-half-neg-control-log-new-reg",
        RIGHT_HALF_ERROR_LEVELS,
        lambda doses, dil, enl: f"percentage-low-r2-1-2-3-{doses}doses-dil{dil}-half-columns-neg-controls-{enl}",
        None,  # no R² figure for right-half
    ),
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
                    leg_loc="lower center",
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
                    leg_loc="upper center",
                    leg_ncol=3,
                    leg_fontsize=8,
                )

                # d_max
                util.plot_barplot_replicate_data(
                    abs_1, abs_2, abs_3, fig_type="", **common_kwargs
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
                        leg_loc="upper left",
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
    paper_kwargs = dict(
        fig_name="-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper",
        fig_dir=fig_dir,
        leg_loc="upper center",
        leg_ncol=3,
        leg_fontsize=8,
    )
    util.plot_barplot_replicate_data(abs_1, abs_2, abs_3, fig_type="", **paper_kwargs)
    util.plot_barplot_replicate_data(rel_1, rel_2, rel_3, fig_type="relic50", **paper_kwargs)
    util.plot_barplot_replicate_data(abs_1, abs_2, abs_3, fig_type="absic50", **paper_kwargs)


def generate_dose_response_figures(cfg: DoseResponseConfig) -> None:
    generate_residuals_figures(cfg)
    generate_ic50_dmax_r2_figures(cfg)


# -----------------------------------------------------------------------
# Stage 3: Curves
# -----------------------------------------------------------------------

# Curve example layouts are derived from benchmark_common.DOSE_RESPONSE_LAYOUT_SPECS
# via dose_response_curve_examples(), so adding a new layout only requires updating
# the central registry.


def generate_example_curves(cfg: DoseResponseConfig) -> None:
    """
    Regenerates example per-compound curve PNGs for each layout type.

    The curves notebook overwrites (compounds, concentrations, replicates) from the
    layout filename for COMPD/PLAID entries, producing different plate_content per
    layout type. This function mirrors that logic explicitly.

    Output PNGs land in cfg.paper_figures_dir (i.e. figures/).
    The os.chdir() approach from the original script is replaced by passing an
    explicit output_dir to dr.plate_curves_after_error.
    """
    cfg.paper_figures_dir.mkdir(parents=True, exist_ok=True)

    slopes = [0.5, 1, 1.5, 2]
    current_e = 50
    expected_noise = 0.01
    my_min_dist = 0

    # Right-half disturbance, strength 0.4 (paper scenario)
    error_nl = 0.4
    error_type = {
        "type": "right-half",
        "error_function": dt.add_errors_to_right_columns_half,
        "error_correction": nrm.normalize_plate_nearest_control,
        "error": error_nl,
    }
    limits = [{"from": 15, "to": 16}]  # bottom row, as in the curves notebook

    for layout_type, layout_dir, layout_file, compounds, concentrations, replicates in dose_response_curve_examples():
        print("layouts:", layout_type)
        try:
            dilution = dilution_for(concentrations)
        except ValueError:
            dilution = 8  # fallback retained from original script

        params = [
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
            for i in range(compounds)
        ]
        df_params = pd.DataFrame.from_dict(params).set_index("compound")
        df_params["abs IC50"] = dr.IC50(
            df_params["b"], df_params["c"], df_params["d"], df_params["e"]
        )
        plate_content = dr.generate_plate_content(
            dose_response_params=params, replicates=replicates
        )

        plate_type_dict = {
            "type": layout_type,
            "dir": layout_dir,
            "error_correction": error_type["error_correction"],
            "requires_layout_update": False,
        }

        for limit in limits:
            dr.plate_curves_after_error(
                layout_dir,
                layout_file,
                plate_content,
                expected_noise,
                error_type["error_function"],
                error_type["error"],
                error_type["error_correction"],
                my_min_dist,
                lose_from_row=limit["from"],
                lose_to_row=limit["to"],
                df_params=df_params,
                plate_type=plate_type_dict,
                compounds=compounds,
                concentrations=concentrations,
                replicates=replicates,
                output_dir=str(cfg.paper_figures_dir),
            )


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

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
