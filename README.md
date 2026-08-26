# Benchmark for testing models for microplate layouts
This is a cleaned version of the benchmarks presented in [COMPD]([link](https://github.com/astra-uu-se/COMPD/tree/main/evaluation_aaai26)) and [PLAID](https://github.com/pharmbio/plaid/tree/main/simulations) articles. The goal is to make the replication straightforward to use and to expand (by adding new layout types and new disturbances), and to make the benchmark itself more comprehensive (systematic generation of plots and tables), with polished, refactored scripts instead of notebooks. 

e.g., now the script generates proper matching control layouts on the same disturbed plate (by making sure that the scale is the same across all the layouts), which makes the comparison between them clear:

<p align="center">
  <img src="figures/plate_compd-controls-rows-error.png" alt="COMPD layout" width="400">
  <img src="figures/plate_plaid-controls-rows-error.png" alt="PLAID layout" width="400">
  <img src="figures/plate_random-controls-rows-error.png" alt="Random layout" width="400">
  <img src="figures/plate_rows-error-base.png" alt="Initial disturbed plate" width="400">
</p>
<p align="center"><em>Three control layouts on the same disturbed plate.</em></p>


# How to use it

To use the benchmark, you need to perform the following steps:

1. Generate the layout files for screening tests, dose-response tests, or both and place them in the folder `layouts\` (you might have to create it). For a quick test, you can unzip the file `layouts.zip` (it contains the generated layout files for border layouts, randomized layouts, PLAID layouts, COMPD 1.00 layouts, and COMPD 1.34 layouts)
2. Edit, if necessary, `benchmark_common.py` to select which layouts you want to compare in screening tests and which layouts you want to compare in dose-response tests. Currently, the file is configured to compare between randomized layouts, PLAID layouts, COMPD 1.00 layouts, and COMPD 1.34 layouts for both sets of tests.
3. Edit, if necessary, `benchmark_disturbances.py` to select which plate disturbances (shapes and strength levels) you want to test the layouts in screening tests and which plate disturbances (shapes and strength levels) you want to test the layouts in dose-response tests
4. Run the Python scripts `run_screening_benchmark.py` and `run_dose_response_benchmark.py`. Each script goes through its own set of stages (perform simulations, then generate the plots, the additional metrics, the LaTeX code for tables, the LaTeX code for the final text). Use the flag `--stage xxx` where `xxx` is the name of the stage to rerun only a specific stage (see the comments in each of the files for specifics).
The `tables` stage of both scripts writes both LaTeX table fragments under `tables/` and the generated supplementary section fragment under `tikz-figures/`, thus no manual LaTeX rewiring is required for the automatically generated benchmark LaTeX file.
5. Once the scripts are executed, you can recompile `detailed-experimental-results-source/0_supplement.tex` to generate the pdf with updated plots and tables.


### Generated LaTeX sections

The table stages of benchmark scripts also regenerate two LaTeX section fragments:

- `detailed-experimental-results-source/tikz-figures/dr_section_auto.tex`
- `detailed-experimental-results-source/tikz-figures/screening_section_auto.tex`

These files assemble the benchmark PNGs and LaTeX tables into the supplementary-material structure. They are generated from the same registry and filename conventions used by the benchmark scripts.

In normal use, users should **not edit these generated `.tex` files or manually add individual benchmark figure/table wrappers to LaTeX**. Instead, regenerate the relevant pipeline stage:

```bash
python run_dose_response_benchmark.py --stage tables
python run_screening_benchmark.py --stage tables
```

The supplementary source imports the generated sections using:

```tex
\input{tikz-figures/dr_section_auto}
\input{tikz-figures/screening_section_auto}
```

Manual/semantic TikZ figures remain separate files in `detailed-experimental-results-source/tikz-figures/`. Those figures are intentionally not overwritten by the benchmark scripts.


## Editing benchmark layouts

Layout metadata is centralised in `benchmark_common.py`. This module is the source of truth for the layouts compared by the dose-response and screening benchmarks, including their display names, ordering, layout-file selection, and benchmark-specific normalisation configuration.

The registry-driven design prevents the same layout information from being duplicated in simulation, plotting, table-generation, and LaTeX code.

### When to edit `benchmark_common.py`

Edit `benchmark_common.py` when you need to add a new layout, remove an existing layout, or change metadata associated with an existing layout, which can include:

- Its human-readable `display_type` name, such as `COMPD`, `PLAID`, or `Random`.
- Its display/plot order.
- The directory or filename pattern used to find its layout matrices.
- The layout-specific normalisation or error-correction function.
- The set of layout comparisons used for boxplots, statistical tables, or figure ordering.
- Layout geometry or control-placement metadata needed by one benchmark pipeline.

Do **not** use `benchmark_common.py` to define a new disturbance family. Disturbance metadata belongs in `benchmark_disturbances.py`.

### Safe editing workflow

1. Make the smallest possible metadata change in the relevant layout specification.
2. Keep `display_type` stable unless you deliberately intend to change CSV contents, figure labels, and table labels. It is not just a cosmetic label. The scripts use it as an identifier written into generated result files.
3. Preserve existing directory paths, filename regexes, and ordering unless the corresponding artifacts have been verified.
4. Run the registry consistency checks indirectly by starting the relevant benchmark script:

   ```bash
   python run_dose_response_benchmark.py --stage figures
   python run_screening_benchmark.py --stage figures
   ```

5. Inspect the resulting figure/table filenames before replacing committed artifacts.

### Important conventions

- The benchmark scripts write the registry `display_type` into result CSVs. Plotting and table utilities validate these names against the registry rather than inferring layout identity from filenames.
- Dose-response and screening have separate layout registries because their plate encoding and normalisation conventions differ.
- The configured layout order is scientific presentation metadata: it controls row order in plots and tables, as well as the order of pairwise statistical comparisons.
- If a layout's name or ordering changes, regenerate the affected figures and tables together. Do not mix artifacts generated from different registry states.

For a new layout, add its metadata once to the relevant registry and let the existing plate-type helper functions supply it to the pipeline. There is no need to add a second hard-coded layout list inside a benchmark script or plotting function, as everything is handled naturally through the registry.

## Editing disturbance scenarios

Disturbance metadata is centralised in `benchmark_disturbances.py`. A disturbance entry describes a named family of simulated plate effects and provides the metadata used by both benchmark pipelines and the generated supplementary LaTeX.

Each disturbance defines, as applicable:

- A stable machine-readable `key`.
- An emphasised display name (`emph_name`) and longer human-readable description (`long_label`).
- Dose-response identifiers, including `dr_id_text`, error type, output-file naming metadata, and dose-response error levels.
- Screening identifiers, including `screening_type` and screening error levels.
- Publication flags: `publish_dr` and `publish_screening`.

The helper functions `dr_scenarios()` and `screening_disturbances()` return only the disturbances marked for publication in the relevant pipeline.

### When to edit `benchmark_disturbances.py`

Edit this file when you need to change:

- A disturbance's presentation label or caption description.
- Which existing disturbance appears in the dose-response or screening supplementary section.
- The mapping between an existing disturbance and its pre-existing scenario identifier.
- The labels used for error strengths, such as the labels displayed as generated-figure columns.
- Publication metadata for a disturbance that already has compatible generated artifacts.

### Preserve stable identifiers

The following fields should be treated as stable identifiers rather than ordinary prose:

- `key`
- `dr_id_text`
- `screening_type`
- Filename/stem/suffix metadata associated with dose-response scenarios
- Numeric error-level values where they are part of existing CSV and PNG filenames

Changing one of these may cause the scripts to look for different CSVs, produce different PNG/table filenames, or leave `% MISSING:` comments in the generated LaTeX sections. If you intentionally change an identifier, regenerate and verify every downstream artifact that uses it.

### Adding a future disturbance family

Adding a genuinely new disturbance requires more than adding a registry row:

1. Add the disturbance metadata and publication flags in `benchmark_disturbances.py`.
2. Ensure the relevant pipeline can resolve the disturbance to the correct simulation function and normalisation strategy.
3. Confirm that its error levels, file naming, and figure/table grouping are represented by existing generic logic.
4. Run the simulation stage for the new definition only when that scientific change is intended.
5. Regenerate figures, tables, and the corresponding auto-generated LaTeX section.
6. Check the generated `.tex` file for `% MISSING:` lines before compiling the supplement.

Do not add per-disturbance plotting or LaTeX wrappers when generic registry metadata is sufficient. Captions, subsection headings, figure grids, and table inclusion should be derived from the registry and the existing generators.

### Publication flags and generated LaTeX

`publish_dr` and `publish_screening` control which disturbance families are emitted by the generated supplementary sections:

- `generate_dr_section_tex(cfg)` writes `tikz-figures/dr_section_auto.tex`.
- `generate_screening_section_tex(cfg)` writes `tikz-figures/screening_section_auto.tex`.

Therefore, changing a publication flag requires regenerating the corresponding `tables` stage. The generators check whether each expected PNG/table artifact exists. Missing artifacts are recorded as LaTeX comments instead of causing a generation failure.
