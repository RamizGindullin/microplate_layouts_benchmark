# COMPD–PLAID Benchmark Artifact Map

This file tracks how experimental scripts and Python libraries produce the figures and tables consumed by the LaTeX supplement in `evaluation_aaai26/detailed-experimental-results-source/`.

Use it as a living reference when refactoring the benchmark (e.g., adding new layouts, cleaning unused code, or adding new metrics).

***

## Conventions

- **Figure ID**: A short, stable label, usually matching the LaTeX fragment name.
- **LaTeX fragment**: Path under `detailed-experimental-results-source/tikz-figures/` (or other `.tex` sources) that consumes images/tables via `\includegraphics` or `\input`.
- **Image/table file(s)**: Paths (relative to `detailed-experimental-results-source/`) for concrete artifacts.
- **Producing script(s)**: The Python script(s) that write those artifacts into the file system.
- **Upstream data/code**: Layout files, CSVs, and library modules used by the producing scripts.
- **Notes**: Any assumptions, coupling, or special conditions to remember when refactoring.

When you extend this document, keep one row per *logical figure or table*, not per PNG/CSV. Group related artifacts (e.g., multiple panels of a single figure) into one row.

***

## Known filename discrepancies (Step 3 open issues)

These mismatches were discovered during the Step 3 plotting-bug investigation. They must be resolved before the LaTeX supplement compiles cleanly from fresh script output.

| Artifact | LaTeX / repo expects | Script currently generates | Status |
|----------|---------------------|---------------------------|--------|
| Group 1 `dose_response__paper` — absolute IC50 panel | `figures/dose-response-absic50-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` | `figures/dose-response-absic50-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png` | ❌ `fig_name` substring mismatch (`right-half` vs `half-columns`) |
| Group 1 `dose_response__paper` — d_diff panel | `figures/dose-response-d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` (confirmed in repo) | `figures/dose-response-d_diff-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png` | ❌ same `fig_name` mismatch |

**Fix:** in `run_dose_response_benchmark.py`, use `fig_name="-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper"` for both the `d_diff` and `absic50` paper-figure calls, matching the filenames actually present in the COMPD repo.

***

## Group 0 – Main paper artifacts

These are the artifacts referenced from `0b_figures_tables.tex`, i.e. the material used directly in the main article rather than only in the long supplement.

### Mapping table

| Group | Figure ID | LaTeX entry point | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|-------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------|----------------------|-------|
| 0 | `dose_response__paper_curves` | `0b_figures_tables.tex` → `\input{_COMPD anonymized/tikz-figures/dose_response__paper_curves}` | Main-paper dose–response example figure (compound 1) for Random, PLAID, and COMPD under strong bowl-shaped effects. | `figures/plate_layout_rand_02.npy_compound_1-right-half.png`, `figures/plate_layout_20-12-8-3_01.npy_compound_1-right-half.png`, `figures/plate_layout_40-12-8-3_01.npy_compound_1-right-half.png` | `run_dose_response_benchmark.py` | Dose-response simulations; layouts from random/PLAID/COMPD; `libraries/dose_response.py`, `libraries/disturbances.py`, `libraries/normalization.py`, `libraries/utilities.py` | Paper-level entry point into artifacts already consumed by Group 1. Concrete file references are in `tikz-figures/dose_response__paper_curves.tex`. |
| 0 | `tab:screening-pr_10-10-0.2` | `0b_figures_tables.tex` (inline LaTeX table, no separate fragment) | Main-paper summary table of ROC-AUC and PR-AUC mean ± std. dev. across hit rates 1%, 5%, 10%, 20%, 30%, 40% for Random, PLAID, COMPD under very strong bowl-shaped effects. | No generated file currently; values are hardcoded inline in `0b_figures_tables.tex`. | None currently — **hardcoded LaTeX table**. | Should be auto-generated from the same screening ROC/PR result CSVs used for Group 5 ROC/PR figures. | 🚨 High-priority Step 4 target: can silently drift from benchmark results. Should become a `\input`-ed generated `.tex` file. |
| 0 | `a_figure_controls` | `0b_figures_tables.tex` → `\input{tikz-figures/a_figure_controls}` | Main-paper control-placement illustration for Random, PLAID, and COMPD layouts. | `figures/plate_random-controls-rows-error.png`, `figures/plate_plaid-controls-rows-error.png`, `figures/plate_compd-controls-rows-error.png` | Layout-generation or layout-visualization script(s), not the main experiment scripts. | Layout-generation code (`create_compd_layouts.py`, `create_compd_layouts_dose_response.py`, `generate_layouts_utilities.py`); possibly `libraries/utilities.py`. | ⚠️ Source script not yet confirmed. Concrete PNG filenames are referenced in `tikz-figures/a_figure_controls.tex`. |

***

## Group 1 – Dose–response paper figures

These are the dose–response figures that appear in the main paper and the supplement.

### Notes on `plot_barplot_replicate_data` filename convention

For the three barplot metrics in this group:

- **`relic50`** figures: `fig_type="relic50"`, `fig_name="-1-2-3-{doses}doses-{dil}-{disturbance}_{suffix}"` → filename `dose-response-relic50-1-2-3-…png`
- **`absic50`** figures: `fig_type="absic50"`, same `fig_name` convention → filename `dose-response-absic50-1-2-3-…png`
- **`d_diff`** figures: `fig_type=""`, `fig_name="d_diff-1-2-3-{doses}doses-{dil}-{disturbance}_{suffix}"` → filename `dose-response-d_diff-1-2-3-…png`

The `d_diff` prefix is part of `fig_name`, **not** derived from `fig_type`. This means the function must **not** mutate `fig_type` internally (Bug 3 fix from Step 3).

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 1 | `dose_response__paper` | `tikz-figures/dose_response__paper.tex` | Main paper dose–response figure. Three rows: (1) example curves (compounds 1 and 4); (2) residuals + d_diff; (3) relic50 + absic50. | **Plate curves (top row)**: `figures/plate_layout_rand_02.npy_compound_4-right-half.png`, `figures/plate_layout_20-12-8-3_01.npy_compound_4-right-half.png`, `figures/plate_layout_40-12-8-3_01.npy_compound_4-right-half.png`  -   **Residuals**: `figures/residuals-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png`  -   **d_diff**: `figures/dose-response-d_diff-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png`  -   **relic50**: `figures/dose-response-relic50-1-2-3-8doses-dil8-half-columns-neg-controls-0.4_paper.png`  -   **absic50** *(see discrepancy table above)*: `figures/dose-response-absic50-1-2-3-8doses-dil8-right-half-neg-controls-0.4_paper.png` | `run_dose_response_benchmark.py` | **Layouts**: `plate_layout_rand_02.npy`, `plate_layout_20-12-8-3_01.npy`, `plate_layout_40-12-8-3_01.npy`  -   **CSV metrics**: `absoluteic50data-*-8doses-dil8-*.csv`, `relativeic50data-*-8doses-dil8-*.csv`, `residuals-*-8doses-dil8-*.csv` under `generated-data/`  -   **Libraries**: `libraries/dose_response.py`, `libraries/normalization.py`, `libraries/disturbances.py`, `libraries/utilities.py` | `_paper.png` suffix = smaller figure layout tuned for the main paper. The `d_diff` and `absic50` panels use `fig_name` substrings with `right-half`; the `relic50` and `residuals` panels use `half-columns`. See discrepancy table above. |
| 1 | `dose_response_dr_curves` | `tikz-figures/dose_response_dr_curves.tex` | Supplementary example dose–response curves for compounds 1, 4, 9 across all 3 layouts. | **Compound 1**: `figures/plate_layout_rand_02.npy_compound_1-right-half.png`, `figures/plate_layout_20-12-8-3_01.npy_compound_1-right-half.png`, `figures/plate_layout_40-12-8-3_01.npy_compound_1-right-half.png`  -   **Compound 4**: `…_compound_4-right-half.png` (×3)  -   **Compound 9**: `…_compound_9-right-half.png` (×3) | `run_dose_response_benchmark.py` | Same layouts and libraries as `dose_response__paper`. Experiments: underlying plate simulations created by dose-response experiment script(s). | Reuses the same plate instances as `dose_response__paper` top row. |

***

## Group 2 – Dose–response disturbance figures (d_max)

These figures summarise mean absolute difference between expected and obtained d_max parameter under different plate-effect types and strengths.

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 2 | `dose_response_d_dist_bowl_no_neg` | `tikz-figures/dose_response_d_dist_bowl_no_neg.tex` | Mean absolute d_max difference vs doses (6, 8, 12) and bowl-shaped plate-effect strength (0.055 vs 0.085), **without** negative controls in the fit. | `figures/dose-response-d_diff-1-2-3-6doses-dil18-bowl-0.055.png`, `figures/dose-response-d_diff-1-2-3-6doses-dil18-bowl-0.085.png`, `figures/dose-response-d_diff-1-2-3-8doses-dil8-bowl-0.055.png`, `figures/dose-response-d_diff-1-2-3-8doses-dil8-bowl-0.085.png`, `figures/dose-response-d_diff-1-2-3-12doses-dil4-bowl-0.055.png`, `figures/dose-response-d_diff-1-2-3-12doses-dil4-bowl-0.085.png` | `run_dose_response_benchmark.py` | CSV metrics for (6, 8, 12 doses) × (0.055, 0.085) bowl strengths, no-neg variant  -   `libraries/dose_response.py`, `normalization.py`, `disturbances.py`, `utilities.py` | `fig_type=""`, `fig_name="d_diff-1-2-3-{doses}doses-{dil}-bowl-{strength}"` per call. Good Step 4 candidate for aggregated LaTeX tables. |
| 2 | `dose_response_d_dist_bowl_neg` | `tikz-figures/dose_response_d_dist_bowl_neg.tex` | Mean absolute d_max difference under bowl-shaped effects with **4 negative controls** in the fit. Same doses/strength grid. | `figures/dose-response-d_diff-1-2-3-6doses-dil18-bowl-neg-controls-0.055.png`, `figures/dose-response-d_diff-1-2-3-6doses-dil18-bowl-neg-controls-0.085.png`, `figures/dose-response-d_diff-1-2-3-8doses-dil8-bowl-neg-controls-0.055.png`, `figures/dose-response-d_diff-1-2-3-8doses-dil8-bowl-neg-controls-0.085.png`, `figures/dose-response-d_diff-1-2-3-12doses-dil4-bowl-neg-controls-0.055.png`, `figures/dose-response-d_diff-1-2-3-12doses-dil4-bowl-neg-controls-0.085.png` | `run_dose_response_benchmark.py` | Analogous CSV families with 4 negative controls included in the LL4 fit. | Directly comparable to `dose_response_d_dist_bowl_no_neg` to isolate the impact of including negatives. |
| 2 | `dose_response_d_dist_column_neg` | `tikz-figures/dose_response_d_dist_column_neg.tex` | Mean absolute d_max under **column-wise linear** right-half plate effects, 4 negative controls in fit. Strengths 0.2 vs 0.4. | `figures/dose-response-d_diff-1-2-3-6doses-dil18-half-columns-neg-controls-0.2.png`, `figures/dose-response-d_diff-1-2-3-6doses-dil18-half-columns-neg-controls-0.4.png`, `figures/dose-response-d_diff-1-2-3-8doses-dil8-half-columns-neg-controls-0.2.png`, `figures/dose-response-d_diff-1-2-3-8doses-dil8-half-columns-neg-controls-0.4.png`, `figures/dose-response-d_diff-1-2-3-12doses-dil4-half-columns-neg-controls-0.2.png`, `figures/dose-response-d_diff-1-2-3-12doses-dil4-half-columns-neg-controls-0.4.png` | `run_dose_response_benchmark.py` | d_max CSVs for column-wise effects (half-columns) at strengths 0.2 and 0.4. | Complements bowl figures with a different error structure. |
| 2 | `dose_response_relic50_bowl_no_neg` | `tikz-figures/dose_response_relic50_bowl_no_neg.tex` | Relative IC50 mean absolute log10 difference, bowl effects, no negatives. | `figures/dose-response-relic50-1-2-3-6doses-dil18-bowl-0.055.png`, `…-bowl-0.085.png`, `…-8doses-dil8-bowl-0.055.png`, `…-8doses-dil8-bowl-0.085.png`, `…-12doses-dil4-bowl-0.055.png`, `…-12doses-dil4-bowl-0.085.png` | `run_dose_response_benchmark.py` | `fig_type="relic50"`. | |
| 2 | `dose_response_relic50_bowl_neg` | `tikz-figures/dose_response_relic50_bowl_neg.tex` | Relative IC50, bowl effects, 4 negatives in fit. | `figures/dose-response-relic50-1-2-3-6doses-dil18-bowl-neg-controls-0.055.png` … (×6) | `run_dose_response_benchmark.py` | | |
| 2 | `dose_response_relic50_column_neg` | `tikz-figures/dose_response_relic50_column_neg.tex` | Relative IC50, column-wise effects, 4 negatives in fit. Strengths 0.2, 0.4. | `figures/dose-response-relic50-1-2-3-6doses-dil18-half-columns-neg-controls-0.2.png` … (×6) | `run_dose_response_benchmark.py` | | |
| 2 | `dose_response_absic50_bowl_no_neg` | `tikz-figures/dose_response_absic50_bowl_no_neg.tex` | Absolute IC50 mean difference, bowl effects, no negatives. | `figures/dose-response-absic50-1-2-3-6doses-dil18-bowl-0.055.png` … (×6) | `run_dose_response_benchmark.py` | `fig_type="absic50"`. | |
| 2 | `dose_response_absic50_bowl_neg` | `tikz-figures/dose_response_absic50_bowl_neg.tex` | Absolute IC50, bowl effects, 4 negatives. | `figures/dose-response-absic50-1-2-3-6doses-dil18-bowl-neg-controls-0.055.png` … (×6) | `run_dose_response_benchmark.py` | | |
| 2 | `dose_response_absic50_column_neg` | `tikz-figures/dose_response_absic50_column_neg.tex` | Absolute IC50, column-wise effects, 4 negatives. Strengths 0.2, 0.4. | `figures/dose-response-absic50-1-2-3-6doses-dil18-half-columns-neg-controls-0.2.png` … (×6) | `run_dose_response_benchmark.py` | | |

***

## Group 3 – Dose-response low-quality curve percentage figures

These figures show the percentage of dose–response curves with R^2 < 0.8.

### Notes on `plot_r2_percentage` filename convention

The function saves to `{fig_dir}percentage-low-r2{fig_name}.png`. The `fig_name` argument starts with `-curves-1-2-3-` followed by the dose/disturbance suffix. The prefix `percentage-low-r2` is hardcoded in the function.

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 3 | `dose_response_percentage_bowl_no_neg` | `tikz-figures/dose_response_percentage_bowl_no_neg.tex` | % low-R² curves, bowl effects, no negatives. | `figures/percentage-low-r2-curves-1-2-3-6doses-dil18-bowl-0.055.png`, `…-0.085.png`, `…-8doses-dil8-bowl-0.055.png`, `…-0.085.png`, `…-12doses-dil4-bowl-0.055.png`, `…-0.085.png` | `run_dose_response_benchmark.py` | Per-plate R² CSVs (no-neg variant)  -   same libraries as Group 2. | |
| 3 | `dose_response_percentage_bowl_neg` | `tikz-figures/dose_response_percentage_bowl_neg.tex` | % low-R² curves, bowl effects, 4 negatives. | `figures/percentage-low-r2-curves-1-2-3-6doses-dil18-bowl-neg-controls-0.055.png` … (×6) | `run_dose_response_benchmark.py` | | |
| 3 | `dose_response_percentage_column_neg` | `tikz-figures/dose_response_percentage_column_neg.tex` | % low-R² curves, column-wise effects, 4 negatives. Strengths 0.2, 0.4. | `figures/percentage-low-r2-curves-1-2-3-6doses-dil18-half-columns-neg-controls-0.2.png` … (×6) | `run_dose_response_benchmark.py` | | |

***

## Group 4 – Screening main paper figure

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 4 | `screening_data__paper` | `tikz-figures/screening_data__paper.tex` | Main screening figure: (a–c) expected vs obtained for 1% hit rate, mild bowl; (d) Z' MSE vs expected; (e) SSMD MSE vs expected; (f–g) PR curves for 1% and 5% hit rates, strong bowl. | `figures/screening-bowl-0.06-10-10-0.99-stdev-3-4-random.png`, `figures/screening-bowl-0.06-10-10-0.99-stdev-3-4-plaid.png`, `figures/screening-bowl-0.06-10-10-0.99-stdev-3-4-compd.png`, `figures/screening-Zfactor-mse-manuscript.png`, `figures/screening-SSMD-mse-manuscript.png`, `figures/PR-10-10-0.2-1.png`, `figures/PR-10-10-0.2-5.png` | `run_screening_benchmark.py` | 40 plates per layout, 10 positives + 10 negatives, 1 replicate; varying hit rates and bowl strengths  -   `libraries/screening.py`, `disturbances.py`, `normalization.py`, `utilities.py` | Top-row panels shared with `screening_data_expected_obtained2` (bowl 0.06). PR 1% and 5% shared with Group 5. |

***

## Group 5 – Screening ROC/PR curves (supplement)

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 5 | `screening_data_roc_strong` | `tikz-figures/screening_data_roc_strong.tex` | ROC curves for hit rates 1, 5, 10, 20, 30, 40% under strong bowl effects (0.2). | `figures/ROC-10-10-0.2-1.png`, `figures/ROC-10-10-0.2-5.png`, `figures/ROC-10-10-0.2-10.png`, `figures/ROC-10-10-0.2-20.png`, `figures/ROC-10-10-0.2-30.png`, `figures/ROC-10-10-0.2-40.png` | `run_screening_benchmark.py` | Per-hit-rate ROC CSVs; `screening.py`, `utilities.py` | |
| 5 | `screening_data_pr_strong` | `tikz-figures/screening_data_pr_strong.tex` | PR curves for hit rates 10, 20, 30, 40% (1% and 5% are in the main paper). | `figures/PR-10-10-0.2-10.png`, `figures/PR-10-10-0.2-20.png`, `figures/PR-10-10-0.2-30.png`, `figures/PR-10-10-0.2-40.png` | `run_screening_benchmark.py` | Precision–recall CSVs per hit rate. | |

***

## Group 6 – Screening SSMD and Z'-factor robustness

These figures examine how MSE of SSMD and Z' factor behave as bowl-shaped plate-effect strength varies.

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 6 | `screening_data_ssmd` | `tikz-figures/screening_data_ssmd.tex` | Grid of SSMD MSE plots across bowl-effect strengths 0.00–0.08. | `figures/screening-SSMD-mse-screening_metrics_data-10-10-0.0-20250623-reviewing.csv.png`, `…-0.01-….png`, …, `…-0.08-….png` | `run_screening_benchmark.py` | `screening_metrics_data-10-10-{strength}-20250623-reviewing.csv`  -   `screening.py`, `utilities.py` | Good Step 4 candidate for a condensed table of MSE vs strength per layout. |
| 6 | `screening_data_z_factor` | `tikz-figures/screening_data_z_factor.tex` | Grid of Z' factor MSE plots across same bowl strengths. | `figures/screening-Zfactor-mse-screening_metrics_data-10-10-0.0-20250623-reviewing.csv.png`, …, `…-0.08-….png` | `run_screening_benchmark.py` | Same CSV files as SSMD. | Ideal for a joint SSMD + Z' summary table. |

***

## Group 7 – Screening expected vs obtained hit counts

These figures compare expected vs obtained hit distributions across Random, PLAID, and COMPD under varying bowl-shaped plate-effect strengths.

### Mapping table

| Group | Figure ID | LaTeX fragment | Logical components | Image/Table files (relative to `detailed-experimental-results-source/`) | Producing script(s) | Upstream data / code | Notes |
|-------|-----------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-------|
| 7 | `screening_data_expected_obtained1` | `tikz-figures/screening_data_expected_obtained1.tex` | Expected vs obtained values, mild bowl (0.03). | `figures/screening-bowl-0.03-10-10-0.99-stdev-3-4-random.png`, `…-plaid.png`, `…-compd.png` | `run_screening_benchmark.py` | 10 positives + 10 negatives, 1% hit rate, bowl 0.03, 40 plates per layout. | |
| 7 | `screening_data_expected_obtained2` | `tikz-figures/screening_data_expected_obtained2.tex` | Expected vs obtained, moderate bowl (0.06). | `figures/screening-bowl-0.06-10-10-0.99-stdev-3-4-random.png`, `…-plaid.png`, `…-compd.png` | `run_screening_benchmark.py` | Same setup, increased bowl strength. | Top-row panels reused in `screening_data__paper`. |
| 7 | `screening_data_expected_obtained3` | `tikz-figures/screening_data_expected_obtained3.tex` | Expected vs obtained, strong bowl (0.08). | `figures/screening-bowl-0.08-10-10-0.99-stdev-3-4-random.png`, `…-plaid.png`, `…-compd.png` | `run_screening_benchmark.py` | Same setup, strongest bowl distortion. | |

***

## Step roadmap
