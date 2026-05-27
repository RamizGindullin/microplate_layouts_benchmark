# COMPD–PLAID Benchmark Artifact Map

This file tracks how experimental scripts and Python libraries produce the figures and
tables consumed by the LaTeX supplement and main paper in
`detailed-experimental-results-source/`.

Use it as a living reference when refactoring the benchmark (e.g., adding new layouts,
cleaning unused code, or adding new metrics).

---

## Conventions

- **Figure ID**: A short, stable label, usually matching the LaTeX fragment name.
- **LaTeX fragment**: Path under `detailed-experimental-results-source/tikz-figures/`
  (or other `.tex` sources) that consumes images/tables via `\includegraphics` or
  `\input`.
- **Image/table file(s)**: Paths relative to the **repository root**. Two locations
  matter:
  - `figures/` — images directly referenced from main-paper LaTeX fragments and from
    `tikz-figures/*.tex` fragments via `\includegraphics{../figures/…}`.
  - `generated-plots/…` — supplement images produced by the run scripts; these must
    be **copied** (or symlinked) into
    `detailed-experimental-results-source/figures/` before LaTeX compilation (see the
    *Output routing* section below).
  - `latex-tables/` — auto-generated `.tex` table fragments; these must be copied into
    `detailed-experimental-results-source/latex-tables/` before compilation.
- **Producing script(s)**: The Python script(s) that write those artifacts.
- **Upstream data/code**: Layout files, CSVs, and library modules used by the producing
  scripts.
- **Notes**: Any assumptions, coupling, or special conditions.

One row per *logical figure or table*, not per PNG/CSV.  Group related artifacts (e.g.
multiple panels of a single figure) into one row.

---

## Output routing

The run scripts write into three top-level output trees:

| Script output root | Contents | Copy target inside `detailed-experimental-results-source/` |
|---|---|---|
| `figures/` | Main-paper PNGs (control layouts, example curves) | `figures/` (already there — no copy needed if repo root IS the LaTeX working dir) |
| `generated-plots/dose-response-supplement/` | Supplement dose–response PNGs (Groups 1–3) | `figures/` |
| `generated-plots/screening-supplement/` | Supplement screening PNGs (Groups 4–5, 7) + ROC/PR | `figures/` |
| `generated-plots/quality-assessment-metrics/` | SSMD / Z′ MSE PNGs (Group 6) | `figures/` |
| `latex-tables/` | Auto-generated `.tex` table fragments | `latex-tables/` |

The LaTeX sources reference figures with paths like `../figures/…` relative to the
`detailed-experimental-results-source/` directory.  The simplest workflow is to run the
scripts from the repository root and ensure the directory
`detailed-experimental-results-source/figures/` is a symlink (or copy target) pointing
at both `figures/` (for main-paper PNGs) and the appropriate `generated-plots/…`
subdirectory (for supplement PNGs).

---

## Known filename discrepancies — RESOLVED (Step 3)

All mismatches discovered during the Step 3 investigation are now fixed in the scripts.

| Artifact | LaTeX / repo expects | Resolved status |
|---|---|---|
| Group 1 `dose_response__paper` — absolute IC50 panel | `figures/dose-response-absic50-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` | ✅ Fixed: `run_dose_response_benchmark.py` now uses `fig_name="-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper"` with `fig_type="absic50"` |
| Group 1 `dose_response__paper` — d_diff panel | `figures/dose-response-d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` | ✅ Fixed: `fig_name="d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper"` with `fig_type=""` |
| Group 6 metrics PNGs — date segment in filename | e.g. `screening_metrics_data-10-10-0.01-20250623-reviewing.csv.png` (date pinned to original run) | ✅ Fixed: `ScreeningConfig.metrics_date_tag = "20250623"` added to `run_screening_benchmark.py`; `run_metrics_simulation` now passes `run_tag=f"{cfg.metrics_date_tag}-{id_text}"` to `sc.test_quality_assessment_metrics`. No change to `libraries/screening.py` — the `run_tag` parameter already existed there. |

---

## Group 0 – Main paper artifacts

Artifacts referenced directly from `0b_figures_tables.tex`.

### Mapping table

| Group | Figure ID | LaTeX entry point | Logical components | Image/Table files (repo-root-relative) | Script output path | Producing script(s) | Upstream data / code | Notes |
|---|---|---|---|---|---|---|---|---|
| 0 | `dose_response__paper_curves` | `0b_figures_tables.tex` → `\input{_COMPD anonymized/tikz-figures/dose_response__paper_curves}` | Main-paper example dose–response curves (compound 1) for Random, PLAID, COMPD under strong bowl-shaped effects. | `figures/plate_layout_rand_02.npy_compound_1-right-half.png`, `figures/plate_layout_20-12-8-3_01.npy_compound_1-right-half.png`, `figures/plate_layout_40-12-8-3_01.npy_compound_1-right-half.png` | Written directly to `figures/` (cfg.paper_figures_dir) | `run_dose_response_benchmark.py` → `generate_example_curves` | Layout `.npy` files from `layouts/`; `libraries/dose_response.py`, `disturbances.py`, `normalization.py`, `utilities.py` | Compound 1 files are shared with Group 1 `dose_response_dr_curves`. No copy step needed — files land in `figures/` directly. |
| 0 | `tab:screening-pr_10-10-0.2` | `0b_figures_tables.tex` (replace hardcoded table with `\input{latex-tables/screening_pr_10-10-0.2}`) | Main-paper summary table: ROC-AUC and PR-AUC mean ± std across hit rates 1%, 5%, 10%, 20%, 30%, 40% for Random, PLAID, COMPD under very strong bowl effects. | `latex-tables/screening_pr_10-10-0.2.tex` | Written to `latex-tables/` (cfg.latex_tables_dir) | `run_screening_benchmark.py` → `generate_auc_latex_table` | Same residuals CSVs as Group 5 ROC/PR figures (`screening-residuals-10-10-0.2-pna-*`); `libraries/screening.py`, `utilities.py` | ✅ No longer hardcoded. Values now computed from the same CSVs as the ROC/PR plots, averaging across all batches using `roc_auc_score` / `average_precision_score`. Must copy `latex-tables/screening_pr_10-10-0.2.tex` → `detailed-experimental-results-source/latex-tables/`. Replace the hardcoded table in `0b_figures_tables.tex` with `\input{latex-tables/screening_pr_10-10-0.2}`. |
| 0 | `a_figure_controls` | `0b_figures_tables.tex` → `\input{tikz-figures/a_figure_controls}` | Main-paper control-placement illustration for Random, PLAID, COMPD under a row-wise linear disturbance. | `figures/plate_random-controls-rows-error.png`, `figures/plate_plaid-controls-rows-error.png`, `figures/plate_compd-controls-rows-error.png` | Written directly to `figures/` | `run_screening_benchmark.py` → `generate_control_layout_figures` | Layout files from `benchmark_common.SCREENING_LAYOUT_SPECS` (`.control_example_file`); `libraries/utilities.plot_plate`; `libraries/disturbances.add_linear_errors_to_upper_rows_half`; `libraries/screening.fill_plate` | ✅ Confirmed. Filenames match `tikz-figures/a_figure_controls.tex` references. No copy step needed — files land in `figures/` directly. |

---

## Group 1 – Dose–response paper figures

Figures appearing in both the main paper and the supplement.

### Notes on filename conventions

- **`relic50`**: `fig_type="relic50"`, `fig_name="-1-2-3-{doses}doses-dil{dil}-{disturbance}"` → `dose-response-relic50-1-2-3-….png`
- **`absic50`**: `fig_type="absic50"`, same `fig_name` → `dose-response-absic50-1-2-3-….png`
- **`d_diff`**: `fig_type=""`, `fig_name="d_diff-1-2-3-{doses}doses-dil{dil}-{disturbance}"` → `dose-response-d_diff-1-2-3-….png`
- **Paper variants** (`_paper` suffix): special explicit calls in `generate_ic50_dmax_r2_figures` using `right-half` or `half-columns` substrings as required (see resolved discrepancy table above).
- All Group 1–3 supplement PNGs land in `generated-plots/dose-response-supplement/`; paper-variant PNGs land in `figures/`.

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (repo-root-relative) | Script output path | Producing script(s) | Upstream data / code | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | `dose_response__paper` | `tikz-figures/dose_response__paper.tex` | Main-paper dose–response figure. Three rows: (1) example curves (compounds 1 and 4); (2) residuals + d_diff; (3) relic50 + absic50. | **Top row**: `figures/plate_layout_rand_02.npy_compound_4-right-half.png`, `figures/plate_layout_20-12-8-3_01.npy_compound_4-right-half.png`, `figures/plate_layout_40-12-8-3_01.npy_compound_4-right-half.png` · **Residuals**: `figures/dose-response-residuals-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png` · **d_diff**: `figures/dose-response-d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` · **relic50**: `figures/dose-response-relic50-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png` · **absic50**: `figures/dose-response-absic50-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` | `figures/` (cfg.paper_figures_dir) | `run_dose_response_benchmark.py` → `generate_residuals_figures` (paper call) + `generate_ic50_dmax_r2_figures` (paper block) + `generate_example_curves` | Layout `.npy` files; residuals/IC50 CSVs under `generated-data/dose-response/`; `libraries/dose_response.py`, `normalization.py`, `disturbances.py`, `utilities.py` | `_paper.png` = smaller layout tuned for the main paper. `d_diff` and `absic50` use `right-half`; `relic50` and `residuals` use `half-columns`. All discrepancies resolved. |
| 1 | `dose_response_dr_curves` | `tikz-figures/dose_response_dr_curves.tex` | Supplementary example curves for compounds 1, 4, 9 across all 3 layouts. | **Cpd 1**: `figures/plate_layout_rand_02.npy_compound_1-right-half.png`, `figures/plate_layout_20-12-8-3_01.npy_compound_1-right-half.png`, `figures/plate_layout_40-12-8-3_01.npy_compound_1-right-half.png` · **Cpd 4**: same pattern `_compound_4-right-half.png` (×3) · **Cpd 9**: `_compound_9-right-half.png` (×3) | `figures/` (cfg.paper_figures_dir) | `run_dose_response_benchmark.py` → `generate_example_curves` | Same layouts and libraries as `dose_response__paper`. | Compound 1 files shared with Group 0 `dose_response__paper_curves`. |

---

## Group 2 – Dose–response disturbance figures (d_max / IC50)

Mean absolute differences for d_max, relic50, absic50 parameters under different plate-effect types and strengths.
All PNGs land in `generated-plots/dose-response-supplement/`.

### Mapping table

| Group | Figure ID | LaTeX fragment | Image/Table files (repo-root-relative, abbreviated) | Producing script(s) | Notes |
|---|---|---|---|---|---|
| 2 | `dose_response_d_dist_bowl_no_neg` | `tikz-figures/dose_response_d_dist_bowl_no_neg.tex` | `generated-plots/dose-response-supplement/dose-response-d_diff-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` → `generate_ic50_dmax_r2_figures` (bowl, no-neg scenario) | Step 4 candidate for aggregated LaTeX table. |
| 2 | `dose_response_d_dist_bowl_neg` | `tikz-figures/dose_response_d_dist_bowl_neg.tex` | `generated-plots/dose-response-supplement/dose-response-d_diff-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-neg-controls-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_d_dist_column_neg` | `tikz-figures/dose_response_d_dist_column_neg.tex` | `generated-plots/dose-response-supplement/dose-response-d_diff-1-2-3-{6,8,12}doses-dil{18,8,4}-half-columns-neg-controls-{0.2,0.4}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_relic50_bowl_no_neg` | `tikz-figures/dose_response_relic50_bowl_no_neg.tex` | `generated-plots/dose-response-supplement/dose-response-relic50-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_relic50_bowl_neg` | `tikz-figures/dose_response_relic50_bowl_neg.tex` | `generated-plots/dose-response-supplement/dose-response-relic50-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-neg-controls-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_relic50_column_neg` | `tikz-figures/dose_response_relic50_column_neg.tex` | `generated-plots/dose-response-supplement/dose-response-relic50-1-2-3-{6,8,12}doses-dil{18,8,4}-half-columns-neg-controls-{0.2,0.4}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_absic50_bowl_no_neg` | `tikz-figures/dose_response_absic50_bowl_no_neg.tex` | `generated-plots/dose-response-supplement/dose-response-absic50-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_absic50_bowl_neg` | `tikz-figures/dose_response_absic50_bowl_neg.tex` | `generated-plots/dose-response-supplement/dose-response-absic50-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-neg-controls-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 2 | `dose_response_absic50_column_neg` | `tikz-figures/dose_response_absic50_column_neg.tex` | `generated-plots/dose-response-supplement/dose-response-absic50-1-2-3-{6,8,12}doses-dil{18,8,4}-half-columns-neg-controls-{0.2,0.4}.png` (6 files) | `run_dose_response_benchmark.py` | |

---

## Group 3 – Dose–response low-quality curve percentage figures

Percentage of dose–response curves with R² < 0.8.
Function `plot_r2_percentage` saves to `{fig_dir}percentage-low-r2{fig_name}.png`;
`fig_name` starts with `-curves-1-2-3-`.
All PNGs land in `generated-plots/dose-response-supplement/`.

### Mapping table

| Group | Figure ID | LaTeX fragment | Image/Table files (repo-root-relative, abbreviated) | Producing script(s) | Notes |
|---|---|---|---|---|---|
| 3 | `dose_response_percentage_bowl_no_neg` | `tikz-figures/dose_response_percentage_bowl_no_neg.tex` | `generated-plots/dose-response-supplement/percentage-low-r2-curves-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` → `generate_ic50_dmax_r2_figures` | |
| 3 | `dose_response_percentage_bowl_neg` | `tikz-figures/dose_response_percentage_bowl_neg.tex` | `generated-plots/dose-response-supplement/percentage-low-r2-curves-1-2-3-{6,8,12}doses-dil{18,8,4}-bowl-neg-controls-{0.055,0.085}.png` (6 files) | `run_dose_response_benchmark.py` | |
| 3 | `dose_response_percentage_column_neg` | `tikz-figures/dose_response_percentage_column_neg.tex` | `generated-plots/dose-response-supplement/percentage-low-r2-curves-1-2-3-{6,8,12}doses-dil{18,8,4}-half-columns-neg-controls-{0.2,0.4}.png` (6 files) | `run_dose_response_benchmark.py` | |

---

## Group 4 – Screening main paper figure

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (repo-root-relative) | Script output path | Producing script(s) | Upstream data / code | Notes |
|---|---|---|---|---|---|---|---|---|
| 4 | `screening_data__paper` | `tikz-figures/screening_data__paper.tex` | (a–c) expected vs obtained, 1% hit rate, mild bowl (0.06); (d) Z′ MSE vs expected; (e) SSMD MSE vs expected; (f–g) PR curves 1% and 5% hit rate, strong bowl (0.2). | `generated-plots/screening-supplement/screening-bowl-0.06-10-10-0.99-stdev-3-4-{random,plaid,compd}.png` · `generated-plots/quality-assessment-metrics/screening-Zfactor-mse-manuscript.png` · `generated-plots/quality-assessment-metrics/screening-SSMD-mse-manuscript.png` · `generated-plots/screening-supplement/PR-10-10-0.2-1.png` · `generated-plots/screening-supplement/PR-10-10-0.2-5.png` | `generated-plots/screening-supplement/` and `generated-plots/quality-assessment-metrics/` | `run_screening_benchmark.py` → `generate_screening_panels` + `generate_metrics_plots` + `generate_roc_pr_curves` | 40 plates per layout, 10 pos + 10 neg, 1 replicate; `libraries/screening.py`, `disturbances.py`, `normalization.py`, `utilities.py` | PR 1% and 5% shared with Group 5. "Manuscript" SSMD/Z′ PNGs = the `error≈0.06` CSV rendered with `fig_name="manuscript"`. |

---

## Group 5 – Screening ROC/PR curves (supplement)

All PNGs land in `generated-plots/screening-supplement/`.

### Mapping table

| Group | Figure ID | LaTeX fragment | Image/Table files (repo-root-relative) | Producing script(s) | Notes |
|---|---|---|---|---|---|
| 5 | `screening_data_roc_strong` | `tikz-figures/screening_data_roc_strong.tex` | `generated-plots/screening-supplement/ROC-10-10-0.2-{1,5,10,20,30,40}.png` (6 files) | `run_screening_benchmark.py` → `generate_roc_pr_curves` | ✅ Method change (approved): curves now average across all batches on a shared 200-point grid using `roc_auc_score`. Consistent with `tab:screening-pr_10-10-0.2` table. |
| 5 | `screening_data_pr_strong` | `tikz-figures/screening_data_pr_strong.tex` | `generated-plots/screening-supplement/PR-10-10-0.2-{10,20,30,40}.png` (4 files; 1% and 5% are in the main paper) | `run_screening_benchmark.py` → `generate_roc_pr_curves` | Same method change as ROC. |

---

## Group 6 – Screening SSMD and Z′-factor robustness

Grid of MSE plots as bowl-effect strength varies from 0.00 to 0.08 in steps of 0.01 (9 values).
All PNGs land in `generated-plots/quality-assessment-metrics/`.
CSV inputs live in `generated-data/quality-assessment-metrics/`.

### Filename date pinning

The CSV filenames embed a date-tag segment, e.g.:

```
screening_metrics_data-10-10-0.01-20250623-reviewing.csv
```

The date segment is controlled by **`ScreeningConfig.metrics_date_tag`** (default `"20250623"`)
in `run_screening_benchmark.py`. The producing call is:

```python
sc.test_quality_assessment_metrics(..., run_tag=f"{cfg.metrics_date_tag}-{id_text}")
```

`libraries/screening.py::test_quality_assessment_metrics` already accepts `run_tag`;
**no change was needed there**. To regenerate with a new date, update only
`metrics_date_tag` in `ScreeningConfig`. The PNG filenames mirror the CSV names
(`screening-SSMD-mse-{csv_name}.png`), so they are pinned automatically.

### Mapping table

| Group | Figure ID | LaTeX fragment | Image/Table files (repo-root-relative, abbreviated) | Producing script(s) | Notes |
|---|---|---|---|---|---|
| 6 | `screening_data_ssmd` | `tikz-figures/screening_data_ssmd.tex` | `generated-plots/quality-assessment-metrics/screening-SSMD-mse-screening_metrics_data-10-10-{0.0…0.08}-20250623-reviewing.csv.png` (9 files) | `run_screening_benchmark.py` → `run_metrics_simulation` + `generate_metrics_plots` | Date segment pinned via `cfg.metrics_date_tag = "20250623"`. Step 4 candidate for condensed numerical table. |
| 6 | `screening_data_z_factor` | `tikz-figures/screening_data_z_factor.tex` | `generated-plots/quality-assessment-metrics/screening-Zfactor-mse-screening_metrics_data-10-10-{0.0…0.08}-20250623-reviewing.csv.png` (9 files) | `run_screening_benchmark.py` | Same date pinning. Ideal for a joint SSMD + Z′ summary table (Step 4). |

---

## Group 7 – Screening expected vs obtained hit counts

All PNGs land in `generated-plots/screening-supplement/`.

### Mapping table

| Group | Figure ID | LaTeX fragment | Image/Table files (repo-root-relative) | Producing script(s) | Notes |
|---|---|---|---|---|---|
| 7 | `screening_data_expected_obtained1` | `tikz-figures/screening_data_expected_obtained1.tex` | `generated-plots/screening-supplement/screening-bowl-0.03-10-10-0.99-stdev-3-4-{random,plaid,compd}.png` | `run_screening_benchmark.py` → `generate_screening_panels` | Mild bowl (0.03). |
| 7 | `screening_data_expected_obtained2` | `tikz-figures/screening_data_expected_obtained2.tex` | `generated-plots/screening-supplement/screening-bowl-0.06-10-10-0.99-stdev-3-4-{random,plaid,compd}.png` | `run_screening_benchmark.py` | Moderate bowl (0.06). Top-row panels reused in `screening_data__paper`. |
| 7 | `screening_data_expected_obtained3` | `tikz-figures/screening_data_expected_obtained3.tex` | `generated-plots/screening-supplement/screening-bowl-0.08-10-10-0.99-stdev-3-4-{random,plaid,compd}.png` | `run_screening_benchmark.py` | Strong bowl (0.08). |

---

## Step roadmap

| Step | Description | Status |
|---|---|---|
| 1 | Artifact map | ✅ Done |
| 2 | Consolidate notebooks → `run_dose_response_benchmark.py` + `run_screening_benchmark.py` | ✅ Done |
| 3 | Clean unused code; registry-driven layout variables; output-folder routing | ✅ Done |
| 4 | Add aggregated LaTeX tables for SSMD/Z′ MSE grids and ROC/PR AUC grids | 🔜 Next |
| 5 | Expand disturbance coverage (activate unused types, add new ones) | 🔜 Planned |

---

## Library dependency summary

| Library | Used by | Role |
|---|---|---|
| `libraries/dose_response.py` | `run_dose_response_benchmark.py` | LL4 curve fitting, plate simulation, CSV export, example-curve PNG generation |
| `libraries/screening.py` | `run_screening_benchmark.py` | Screening simulation, SSMD/Z′ metrics |
| `libraries/disturbances.py` | Both run scripts | Plate-effect generation (bowl, row-wise linear, etc.) |
| `libraries/normalization.py` | Both run scripts (via `benchmark_common._default_error_correction`) | LOWESS 2-D plate normalization |
| `libraries/utilities.py` | Both run scripts | All plotting helpers (`plot_barplot_replicate_data`, `plot_r2_percentage`, `plot_screening_plates`, `plot_roc_curves`, `plot_pr_curves`, `plotting_residual_metrics`, `plot_plate`) |
| `benchmark_common.py` | Both run scripts | Layout registries (`DOSE_RESPONSE_LAYOUT_SPECS`, `SCREENING_LAYOUT_SPECS`), shared config helpers |
