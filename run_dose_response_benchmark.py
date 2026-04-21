#!/usr/bin/env python3
"""
run_screening_benchmark.py

Replacement for:
  - screening-experiments.ipynb  (Stage 1: simulation + CSV generation)
  - screening-supplement.ipynb   (Stage 2: figure generation)

Produces exactly the same output files referenced by the LaTeX sources.
Run from the evaluation_aaai26/ directory.

Usage:
  python run_screening_benchmark.py                    # run all stages
  python run_screening_benchmark.py --stage simulate   # CSVs only
  python run_screening_benchmark.py --stage figures    # figures only (CSVs must exist)
  python run_screening_benchmark.py --stage tables     # AUC table only (CSVs must exist)
  python run_screening_benchmark.py --run-tag 20250623-ROC-supplement
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import libraries.disturbances as dt
import libraries.normalization as nrm
import libraries.screening as sc
import libraries.utilities as util


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScreeningConfig:
    # Experiment grid
    neg_pos_controls_list: List[Tuple[int, int]] = field(
        default_factory=lambda: [(8, 8), (10, 10), (20, 10)]
    )
    error_strength_list: List[float] = field(
        default_factory=lambda: [0.06, 0.1, 0.2]
    )
    hit_rate_list: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    )

    # Plate biology
    neg_control_mean: float = 100.0
    neg_stdev: float = 10.0
    pos_stdev: float = 10.0
    # pos_control_mean is iterated over in a range(40, 41, 120) → only value 40
    pos_control_mean: int = 40

    # Batches per condition
    n_batches: int = 10

    # Row-loss sweep: range(1, 4) → lost_rows in {1, 2, 3}
    lost_rows_range: Tuple[int, int] = (1, 4)

    # Directories (relative to evaluation_aaai26/)
    data_dir: Path = Path("generated-data/screening")
    plots_dir: Path = Path("generated-plots/screening-supplement")
    latex_tables_dir: Path = Path("latex-tables")

    # Run tag (controls output filenames; must match what LaTeX expects)
    run_tag: str = "20250623-ROC-supplement"

    # Error types active in the current benchmark
    # (others are commented out in the notebook)
    @property
    def error_types(self):
        return [
            {
                "type": "bowl-nl",
                "error_function": dt.add_bowlshaped_errors_nl,
                "error_correction": nrm.normalize_plate_lowess_2d,
            }
        ]

    @property
    def today_tag(self) -> str:
        return f"-{self.run_tag}"

    def layout_types(self, neg_controls: int, pos_controls: int) -> list:
        nc, pc = neg_controls, pos_controls
        return [
            {
                "type": "random",
                "dir": "layouts/screening_RANDM_layouts/",
                "regex": f"plate_layout_rand_{nc}-{pc}_" + r"(0*)(.+?)\.npy",
                "error_correction": nrm.normalize_plate_lowess_2d,
            },
            {
                "type": "plaid",
                "dir": "layouts/screening_PLAID_layouts/",
                "regex": f"plate_layout_{nc}-{pc}_" + r"(0*)(.+?)\.npy",
                "error_correction": nrm.normalize_plate_lowess_2d,
            },
            {
                "type": "compd",
                "dir": "layouts/screening_COMPD_layouts/",
                "regex": f"plate_layout_{nc}-{pc}_" + r"(0*)(.+?)\.npy",
                "error_correction": nrm.normalize_plate_lowess_2d,
            },
        ]

    def scores_filename(self, pos_controls, neg_controls, error, pna) -> Path:
        name = (
            f"screening_scores_data-{pos_controls}-{neg_controls}"
            f"-{error}-pna-{pna}{self.today_tag}.csv"
        )
        return self.data_dir / name

    def residuals_filename(self, pos_controls, neg_controls, error, pna) -> Path:
        name = (
            f"screening-residuals-{pos_controls}-{neg_controls}"
            f"-{error}-pna-{pna}{self.today_tag}.csv"
        )
        return self.data_dir / name


# ---------------------------------------------------------------------------
# Stage 1 — Simulation
# ---------------------------------------------------------------------------

SCORES_HEADER = [
    "batch", "layout", "error_type", "error", "lost_rows",
    "neg_control_mean", "pos_control_mean", "neg_stdev", "pos_stdev",
    "Zfactor_expected", "SSMD_expected",
    "Zfactor_raw", "SSMD_raw",
    "Zfactor_norm", "SSMD_norm",
]

RESIDUALS_HEADER = [
    "batch", "layout", "error_type", "error", "lost_rows",
    "neg_control_mean", "pos_control_mean", "neg_stdev", "pos_stdev",
    "comp_id", "true_residuals", "expected_result", "obtained_result",
    "activity", "plate_id",
]


def simulate_condition(
    config: ScreeningConfig,
    neg_controls: int,
    pos_controls: int,
    error: float,
    percent_non_active: float,
) -> None:
    """
    Simulate one (neg, pos, error, pna) condition across all layout types,
    batches, and row-loss values.  Writes two CSVs to config.data_dir.
    """
    scores_path = config.scores_filename(pos_controls, neg_controls, error, percent_non_active)
    residuals_path = config.residuals_filename(pos_controls, neg_controls, error, percent_non_active)

    print(f"  Writing: {scores_path.name}")
    print(f"  Writing: {residuals_path.name}")

    plate_types = config.layout_types(neg_controls, pos_controls)

    stop_all = False

    with open(scores_path, "w", newline="") as sf, \
         open(residuals_path, "w", newline="") as rf:

        scores_writer = csv.writer(sf)
        residuals_writer = csv.writer(rf)
        scores_writer.writerow(SCORES_HEADER)
        residuals_writer.writerow(RESIDUALS_HEADER)

        for batch in range(config.n_batches):
            if stop_all:
                break
            print(f"    batch {batch}")

            # The notebook loops range(40, 41, 120) → only pos_control_mean=40
            for pos_control_mean in range(40, 41, 120):
                if stop_all:
                    break

                for plate_type in plate_types:
                    if stop_all:
                        break

                    layout_dir = plate_type["dir"]
                    layouts = sorted(os.listdir(layout_dir))

                    for layout_file in layouts:
                        if stop_all:
                            break

                        match = re.search(plate_type["regex"], layout_file)
                        if match is None:
                            continue

                        layout = np.load(layout_dir + layout_file)
                        neg_control_id = int(np.max(layout))
                        pos_control_id = neg_control_id - 1

                        for et in config.error_types:
                            for lost_rows in range(*config.lost_rows_range):
                                limits = [{"from": 1, "to": lost_rows}]

                                for limit in limits:
                                    # --- ideal plate ---
                                    ideal_plate, activity_layout = sc.fill_plate(
                                        layout,
                                        neg_control_id,
                                        pos_control_id,
                                        config.neg_control_mean,
                                        pos_control_mean,
                                        neg_stdev=config.neg_stdev,
                                        pos_stdev=config.pos_stdev,
                                        percent_non_active=percent_non_active,
                                    )

                                    exp_nc_mean, exp_pc_mean, exp_nc_std, exp_pc_std = (
                                        sc.control_stats(
                                            ideal_plate, layout,
                                            neg_control_id, pos_control_id
                                        )
                                    )
                                    ssmd_expected = sc.ssmd(
                                        exp_nc_mean, exp_pc_mean, exp_nc_std, exp_pc_std
                                    )
                                    zfactor_expected = sc.zfactor(
                                        exp_nc_mean, exp_pc_mean, exp_nc_std, exp_pc_std
                                    )

                                    # --- apply disturbance + row loss ---
                                    plate = et["error_function"](ideal_plate, error)
                                    plate = dt.lose_rows(plate, limit["from"], limit["to"])

                                    # --- raw stats ---
                                    raw_nc_mean, raw_pc_mean, raw_nc_std, raw_pc_std = (
                                        sc.control_stats(
                                            plate, layout,
                                            neg_control_id, pos_control_id
                                        )
                                    )
                                    ssmd_raw = sc.ssmd(
                                        raw_nc_mean, raw_pc_mean, raw_nc_std, raw_pc_std
                                    )
                                    zfactor_raw = sc.zfactor(
                                        raw_nc_mean, raw_pc_mean, raw_nc_std, raw_pc_std
                                    )

                                    # --- normalization ---
                                    remaining_layout = dt.lose_rows(
                                        layout, limit["from"], limit["to"]
                                    )
                                    try:
                                        plate = plate_type["error_correction"](
                                            plate,
                                            remaining_layout,
                                            neg_control_id=neg_control_id,
                                        )
                                    except Exception as exc:
                                        print(
                                            f"      ERROR normalizing {layout_file}: {exc}"
                                        )
                                        stop_all = True
                                        break

                                    # --- normalized stats ---
                                    norm_nc_mean, norm_pc_mean, norm_nc_std, norm_pc_std = (
                                        sc.control_stats(
                                            plate, remaining_layout,
                                            neg_control_id, pos_control_id
                                        )
                                    )
                                    ssmd_norm = sc.ssmd(
                                        norm_nc_mean, norm_pc_mean, norm_nc_std, norm_pc_std
                                    )
                                    zfactor_norm = sc.zfactor(
                                        norm_nc_mean, norm_pc_mean, norm_nc_std, norm_pc_std
                                    )

                                    # --- write scores row ---
                                    scores_writer.writerow([
                                        batch, plate_type["type"], et["type"], error,
                                        lost_rows - 1,
                                        exp_nc_mean, pos_control_mean,
                                        exp_nc_std, exp_pc_std,
                                        zfactor_expected, ssmd_expected,
                                        zfactor_raw, ssmd_raw,
                                        zfactor_norm, ssmd_norm,
                                    ])

                                    # --- write residuals rows ---
                                    ideal_lost = dt.lose_rows(
                                        ideal_plate, limit["from"], limit["to"]
                                    )
                                    res_array = np.power(
                                        np.abs(ideal_lost - plate), 2
                                    ).reshape(-1, 1)
                                    comp_id_array = remaining_layout.reshape(-1, 1)
                                    ideal_array = ideal_plate.reshape(-1, 1)
                                    norm_array = plate.reshape(-1, 1)
                                    activity_array = activity_layout.reshape(-1, 1)

                                    combined = np.hstack([
                                        comp_id_array, res_array,
                                        ideal_array, norm_array, activity_array
                                    ])
                                    df = pd.DataFrame(
                                        combined,
                                        columns=[
                                            "comp_type", "res",
                                            "expected_result", "obtained_result",
                                            "activity",
                                        ],
                                    )
                                    df = df[df.comp_type > 0]
                                    arr = df.to_numpy().T
                                    (_, res_size) = arr.shape

                                    plate_residuals = np.vstack([
                                        np.full(res_size, batch),
                                        np.full(res_size, plate_type["type"]),
                                        np.full(res_size, et["type"]),
                                        np.full(res_size, error),
                                        np.full(res_size, lost_rows - 1),
                                        np.full(res_size, exp_nc_mean),
                                        np.full(res_size, pos_control_mean),
                                        np.full(res_size, exp_nc_std),
                                        np.full(res_size, exp_pc_std),
                                        arr[0],   # comp_id
                                        arr[1],   # true_residuals
                                        arr[2],   # expected_result
                                        arr[3],   # obtained_result
                                        arr[4],   # activity
                                        np.full(res_size, layout_file),
                                    ])
                                    residuals_writer.writerows(plate_residuals.T)


def run_simulations(config: ScreeningConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    for (neg_controls, pos_controls) in config.neg_pos_controls_list:
        print(f"\n(neg_controls={neg_controls}, pos_controls={pos_controls})")
        for error in config.error_strength_list:
            print(f"  error={error}")
            for hit_rate in config.hit_rate_list:
                pna = round(1.0 - hit_rate, 10)
                print(f"    hit_rate={hit_rate:.0%}  pna={pna}")
                simulate_condition(config, neg_controls, pos_controls, error, pna)


# ---------------------------------------------------------------------------
# Stage 2a — Control-layout figures
# (screening-supplement.ipynb cells 4–6)
# ---------------------------------------------------------------------------

def generate_control_figures(config: ScreeningConfig) -> None:
    """
    Generates the three control-layout plate images referenced by
    tikz-figures/a_figure_controls.tex:
      figures/plate_random-controls-rows-error.png
      figures/plate_plaid-controls-rows-error.png
      figures/plate_compd-controls-rows-error.png
    """
    import matplotlib
    matplotlib.use("Agg")

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    layout_specs = [
        (
            "layouts/screening_RANDM_layouts/",
            "plate_layout_rand_10-10_02.npy",
            figures_dir / "plate_random-controls-rows-error.png",
        ),
        (
            "layouts/screening_PLAID_layouts/",
            "plate_layout_10-10_01.npy",
            figures_dir / "plate_plaid-controls-rows-error.png",
        ),
        (
            "layouts/screening_COMPD_layouts/",
            "plate_layout_10-10_01.npy",
            figures_dir / "plate_compd-controls-rows-error.png",
        ),
    ]

    neg_control_mean = 90
    pos_control_mean = 60
    neg_stdev = 2
    pos_stdev = 7

    for layout_dir, layout_file, out_path in layout_specs:
        layout = np.load(layout_dir + layout_file)
        neg_control_id = int(np.max(layout))
        pos_control_id = neg_control_id - 1

        ideal_plate, _ = sc.fill_plate(
            layout, neg_control_id, pos_control_id,
            neg_control_mean, pos_control_mean, neg_stdev, pos_stdev,
        )
        # Apply the same row-error disturbance used in the notebook (upper rows, linear)
        disturbed_plate = dt.add_linear_errors_to_upper_rows_half(ideal_plate, 4)

        control_locations = util.get_controls_layout(layout)
        util.plot_plate(
            disturbed_plate,
            title="",
            mask=np.array(1 - control_locations, dtype=bool),
            filename=str(out_path),
        )
        print(f"  Written: {out_path}")


# ---------------------------------------------------------------------------
# Stage 2b — Expected-vs-obtained screening panels
# (screening-supplement.ipynb cells 8–11)
# ---------------------------------------------------------------------------

def generate_screening_panels(config: ScreeningConfig) -> None:
    """
    Generates the expected-vs-obtained plate panels for several bowl strengths.
    Output: generated-plots/screening-supplement/screening-bowl-<fig_name>-{random,plaid,compd}.png
    """
    fig_dir = str(config.plots_dir) + "/"
    data_dir = str(config.data_dir) + "/"
    tag = config.today_tag

    panels = [
        # (fig_name, residuals_file_error, residuals_file_pna, max_value)
        # Cell 8 — Manuscript Figure 3 a,b,c
        ("0.06-10-10-0.99-stdev-3-4",
         data_dir + f"screening-residuals-10-10-0.1-pna-0.99{tag}.csv",
         450),
        # Cell 9 — Supplement Figure
        ("0.08-10-10-0.99-stdev-3-4",
         data_dir + f"screening-residuals-10-10-0.2-pna-0.99{tag}.csv",
         450),
        # Cell 10 — Supplement (not included)
        ("0.03-10-10-0.99-stdev-3-4",
         data_dir + f"screening-residuals-10-10-0.06-pna-0.99{tag}.csv",
         450),
        # Cell 11 — Supplement (not included)
        ("0.05-10-10-0.99-stdev-3-4",
         data_dir + f"screening-residuals-10-10-0.05-pna-0.99{tag}.csv",
         250),
    ]

    for fig_name, residuals_filename, max_value in panels:
        print(f"  Panel: {fig_name}")
        util.plot_screening_plates(
            residuals_filename,
            fig_name=fig_name,
            fig_dir=fig_dir,
            max_value=max_value,
        )


# ---------------------------------------------------------------------------
# Stage 2c — ROC and PR curves
# (screening-supplement.ipynb cells 12–23)
# ---------------------------------------------------------------------------

def generate_roc_pr_curves(config: ScreeningConfig) -> None:
    """
    Generates ROC and PR curve figures for all hit rates and control configurations.
    Prints AUC table code (roc_table_code / pr_table_code) to stdout for
    the main-paper ROC/PR table.
    """
    fig_dir = str(config.plots_dir) + "/"
    data_dir = str(config.data_dir) + "/"
    tag = config.today_tag

    # Each entry: (residuals_file, fig_name_stem, batch_roc, batch_pr, print_table)
    # batch=None → let util use its default
    roc_pr_specs = [
        # 10-10 controls, strong bowl 0.2 — main paper + supplement
        (f"screening-residuals-10-10-0.2-pna-0.99{tag}.csv", "10-10-0.2-1.png",  6, 6,  True),   # Fig 3f
        (f"screening-residuals-10-10-0.2-pna-0.95{tag}.csv", "10-10-0.2-5.png",  9, 9,  True),   # Fig 3g
        (f"screening-residuals-10-10-0.2-pna-0.9{tag}.csv",  "10-10-0.2-10.png", 0, 0,  True),   # Fig 23a
        (f"screening-residuals-10-10-0.2-pna-0.8{tag}.csv",  "10-10-0.2-20.png", 2, 2,  True),   # Fig 23b
        (f"screening-residuals-10-10-0.2-pna-0.7{tag}.csv",  "10-10-0.2-30.png", 6, 6,  True),   # Fig 23c
        (f"screening-residuals-10-10-0.2-pna-0.6{tag}.csv",  "10-10-0.2-40.png", None, None, True),  # Fig 23d

        # 8-8 controls, bowl 0.1 — supplement Figures 24a–f
        (f"screening-residuals-8-8-0.1-pna-0.99{tag}.csv", "8-8-0.1-1.png",  None, None, False),
        (f"screening-residuals-8-8-0.1-pna-0.95{tag}.csv", "8-8-0.1-5.png",  1,    1,    False),
        (f"screening-residuals-8-8-0.1-pna-0.9{tag}.csv",  "8-8-0.1-10.png", 6,    6,    False),
        (f"screening-residuals-8-8-0.1-pna-0.8{tag}.csv",  "8-8-0.1-20.png", 6,    6,    False),
        (f"screening-residuals-8-8-0.1-pna-0.7{tag}.csv",  "8-8-0.1-30.png", 6,    6,    False),
        (f"screening-residuals-8-8-0.1-pna-0.6{tag}.csv",  "8-8-0.1-40.png", 3,    3,    True),
    ]

    for fname, fig_stem, batch_roc, batch_pr, print_table in roc_pr_specs:
        residuals_path = data_dir + fname
        roc_fig = "ROC-" + fig_stem
        pr_fig  = "PR-"  + fig_stem
        print(f"  ROC/PR: {fig_stem}")

        roc_kwargs = {"batch": batch_roc} if batch_roc is not None else {}
        pr_kwargs  = {"batch": batch_pr}  if batch_pr  is not None else {}

        util.plot_roc_curves(residuals_path, roc_fig, fig_dir, **roc_kwargs)
        util.plot_pr_curves( residuals_path, pr_fig,  fig_dir, **pr_kwargs)

        if print_table:
            util.roc_table_code(residuals_path)
            print("--------------")
            util.pr_table_code(residuals_path)


# ---------------------------------------------------------------------------
# Stage 3 — Auto-generated LaTeX AUC table
# ---------------------------------------------------------------------------

def generate_auc_latex_table(config: ScreeningConfig) -> None:
    """
    Replaces the hardcoded ROC-AUC / PR-AUC table in 0b_figures_tables.tex
    by computing values from the same residuals CSVs used by the ROC/PR plots.

    Output: latex-tables/screening_pr_10-10-0.2.tex
    """
    from sklearn import metrics as skmetrics

    config.latex_tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir = str(config.data_dir) + "/"
    tag = config.today_tag

    hit_rates   = [1, 5, 10, 20, 30, 40]
    pna_values  = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    batches_roc = [6, 9, 0, 2, 6, None]   # match notebook batch selections
    layouts     = ["random", "plaid", "compd"]

    # Accumulate: summary[hit_rate][layout] = {"roc": [...], "pr": [...]}
    summary = {}
    for hit_rate, pna, batch in zip(hit_rates, pna_values, batches_roc):
        fname = f"screening-residuals-10-10-0.2-pna-{pna}{tag}.csv"
        path = data_dir + fname
        if not os.path.exists(path):
            print(f"  SKIP (missing): {fname}")
            continue

        df = pd.read_csv(path)
        if batch is not None:
            df = df[df["batch"] == batch]

        summary[hit_rate] = {}
        for layout in layouts:
            sub = df[df["layout"] == layout]
            if sub.empty:
                summary[hit_rate][layout] = {"roc": float("nan"), "pr": float("nan")}
                continue
            y_true = (sub["activity"] > 0).astype(int)
            y_score = -sub["true_residuals"]   # lower residual → more likely active
            roc_auc = skmetrics.roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else float("nan")
            pr_auc  = skmetrics.average_precision_score(y_true, y_score) if y_true.nunique() > 1 else float("nan")
            summary[hit_rate][layout] = {"roc": roc_auc, "pr": pr_auc}

    # Build LaTeX table
    col_labels = ["Random ROC", "PLAID ROC", "COMPD ROC",
                  "Random PR",  "PLAID PR",  "COMPD PR"]
    lines = [
        r"\begin{tabular}{r" + "c" * 6 + r"}",
        r"\toprule",
        r"Hit rate & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for hit_rate in hit_rates:
        if hit_rate not in summary:
            continue
        row = [f"{hit_rate}\\%"]
        for metric in ["roc", "pr"]:
            for layout in layouts:
                val = summary[hit_rate][layout].get(metric, float("nan"))
                row.append(f"{val:.3f}" if not np.isnan(val) else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    out_path = config.latex_tables_dir / "screening_pr_10-10-0.2.tex"
    out_path.write_text("\n".join(lines))
    print(f"  Written: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Screening benchmark script")
    p.add_argument(
        "--stage",
        choices=["all", "simulate", "figures", "tables"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    p.add_argument(
        "--run-tag",
        default="20250623-ROC-supplement",
        help="Tag embedded in output filenames (must match LaTeX expectations)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = ScreeningConfig(run_tag=args.run_tag)

    config.plots_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "simulate"):
        print("\n=== Stage 1: Simulation ===")
        run_simulations(config)

    if args.stage in ("all", "figures"):
        print("\n=== Stage 2a: Control layout figures ===")
        generate_control_figures(config)

        print("\n=== Stage 2b: Expected-vs-obtained panels ===")
        generate_screening_panels(config)

        print("\n=== Stage 2c: ROC / PR curves ===")
        generate_roc_pr_curves(config)

    if args.stage in ("all", "tables"):
        print("\n=== Stage 3: AUC LaTeX table ===")
        generate_auc_latex_table(config)

    print("\nDone.")


if __name__ == "__main__":
    main()