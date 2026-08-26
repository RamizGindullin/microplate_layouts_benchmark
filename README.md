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


> [!TIP]
> ## Start here
>
> **I want to reproduce existing figures and tables**
>
> ```bash
> python run_dose_response_benchmark.py --stage figures
> python run_dose_response_benchmark.py --stage tables
>
> python run_screening_benchmark.py --stage figures
> python run_screening_benchmark.py --stage metrics
> python run_screening_benchmark.py --stage tables
> ```
>
> **I want to change a layout**
>
> Read [Editing benchmark layouts](#editing-benchmark-layouts-registry), then run the affected simulation, figure, metrics, and table stages.
>
> **I want to change a disturbance or its labels**
>
> Read [Editing disturbance scenarios](#editing-disturbance-scenarios).
>
> **I want to import a layout from another generator**
>
> Read [Generating and importing layout matrices](#generating-and-importing-layout-matrices), especially the layout encoding and filename contract.


## Contents

- [How to use it](#how-to-use-it)
  - [Generated LaTeX sections](#generated-latex-sections)
- [Installation](#installation)
- [Running the benchmark pipelines](#running-the-benchmark-pipelines)
  - [Dose-response pipeline](#dose-response-pipeline)
  - [Screening pipeline](#screening-pipeline)
  - [Recommended workflows](#recommended-workflows)
- [Repository data flow](#repository-data-flow)
- [Editing benchmark layouts](#editing-benchmark-layouts)
- [Generating and importing layout matrices](#generating-and-importing-layout-matrices)
  - [Reference layout-generation material](#reference-layout-generation-material)
  - [PLAID-compatible well encoding and replicate semantics](#plaid-compatible-well-encoding-and-replicate-semantics)
  - [Filename and directory contract](#filename-and-directory-contract)
  - [Layout families currently compared](#layout-families-currently-compared)
  - [Safe import workflow](#safe-import-workflow)
- [Editing disturbance scenarios](#editing-disturbance-scenarios)
  - [Currently enabled disturbance scenarios](#currently-enabled-disturbance-scenarios)
  - [Dose-response](#dose-response)
  - [Screening](#screening)
- [Reproducibility and artifact policy](#reproducibility-and-artifact-policy)
- [Troubleshooting](#troubleshooting)
- [Citation and licence](#citation-and-licence)


# How to use the benchmark

To use the benchmark, you need to perform the following steps:

1. Generate the layout files for screening tests, dose-response tests, or both and place them in the folder `layouts\` (you might have to create it). For a quick test, you can unzip the file `layouts.zip` (it contains the generated layout files for border layouts, randomized layouts, PLAID layouts, COMPD 1.00 layouts, and COMPD 1.34 layouts)
2. Edit, if necessary, `benchmark_common.py` to select which layouts you want to compare in screening tests and which layouts you want to compare in dose-response tests. Currently, the file is configured to compare between randomized layouts, PLAID layouts, and COMPD 1.00 layouts for both sets of tests.
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


## Generating and importing layout matrices

The benchmark is **layout-generator agnostic**. It can evaluate layouts generated by COMPD, PLAID, another optimisation model, a graphical tool, or a custom script.

What matters is not the source of a layout, but whether it satisfies the benchmark’s input contract:

1. The layout is stored as a NumPy `.npy` file.
2. Its well encoding follows the PLAID-compatible integer format described below.
3. Its filename and directory match the registered lookup metadata in `benchmark_common.py`.
4. Its dimensions, number of controls, compound/dose/replicate capacity, and normalisation assumptions are compatible with the relevant benchmark configuration.

The benchmark scripts do not invoke a layout solver during normal operation. They load already-generated `.npy` matrices from the registered layout directories.

### Reference layout-generation material

The COMPD repository contains the generation material used for the associated evaluation work:

- [Evaluation layout-generation directory](https://github.com/astra-uu-se/COMPD/tree/main/evaluation_aaai26)
- [MiniZinc `.dzn` input files](https://github.com/astra-uu-se/COMPD/tree/main/evaluation_aaai26/dzn-files)

The `.dzn` files are useful templates for specifying an experiment before generation: plate dimensions, number of experimental items, controls, concentrations, replicates, edge-well policy, and other layout constraints.

They are references and starting points rather than a requirement. A layout generated elsewhere is fully compatible with this benchmark provided that it is exported in the required `.npy` format and registered correctly.

### PLAID-compatible well encoding and replicate semantics

The values in a layout matrix are not arbitrary labels. They define the mapping between physical wells and the benchmark’s simulated assay observations.

The benchmark expects an integer matrix in which every well is one of:

- `0`: an unused/empty well;
- a positive experimental-item identifier;
- the negative-control identifier.

The meaning of a positive experimental-item identifier depends crucially on whether the layout represents technical replicates as repeated labels or as separately encoded wells.

### Experimental conditions versus physical observations

Suppose a dose-response experiment has:

```text
C compounds
D concentrations per compound
R technical replicates per concentration
```

There are then two different counts:

```text
Experimental conditions:       C × D
Physical experimental wells:   C × D × R
```

For example, an experiment with 36 compounds, 6 concentrations, and 2 replicates has:

```text
36 × 6     = 216 compound-concentration conditions
36 × 6 × 2 = 432 physical experimental-well observations
```

A layout generator may encode those two replicate wells in either of two ways.

<details>
<summary><strong>Technical detail: condition-level versus per-well replicate IDs</strong></summary>

#### Condition-level encoding

In a condition-level representation, all replicate wells for the same compound-concentration condition share one integer identifier.

For example, condition 17 may appear twice:

```text
... 17 ... 17 ...
```

This representation has identifiers:

```text
1, 2, ..., C × D
```

and a negative-control identifier of:

```text
C × D + 1
```

For the 36-compound, 6-dose, 2-replicate example:

```text
Experimental IDs:       1 ... 216
Negative-control ID:    217
```

This is a natural representation for a layout generator because it says that both wells belong to the same compound-concentration condition.

However, it is **not directly compatible** with the main dose-response benchmark reader when the benchmark needs each replicate measurement as a separate observation.

#### Per-well encoding

In a per-well representation, every physical experimental well receives its own unique identifier.

For the same example:

```text
Experimental IDs:       1 ... 432
Negative-control ID:    433
```

The two replicate wells for condition 17 no longer share the value `17`. Instead, they receive distinct IDs, for example:

```text
condition 17, replicate 1  -> 17
condition 17, replicate 2  -> 233
```

The exact numeric assignment matters because the benchmark maps these identifiers back to the simulated `plate_content` observations.
</details>

### Why unique per-well IDs matter

The dose-response benchmark uses the layout matrix to collect plate values into a one-dimensional result vector. Its collection logic effectively uses the layout value as the destination index for a well’s measurement.

If two wells use the same positive identifier, the later encountered well replaces the earlier value:

```python
results[neg_control - cell - 1] = plate[row_index][col_index]
```

Therefore, if replicated experimental wells retain the same condition-level ID, one replicate can overwrite another during result collection.

This is not a minor averaging difference. It can cause the benchmark to:

- retain only one observation for a replicated condition;
- lose replicate information before curve fitting;
- produce a result vector shorter than the expected plate-content vector;
- associate measurements with the wrong compound/concentration/replicate observation; or
- fail the explicit layout/result-length compatibility checks.

For this reason, the main dose-response path requires an effectively unique positive identifier for every physical experimental well.

### COMPD compatibility conversion

COMPD-generated layouts use the natural condition-level encoding described above: all physical wells that are technical replicates of the same compound-concentration condition use the same ID. This was a result of a misunderstanding during the development of the benchmark for the COMPD paper.

To support these layouts, the benchmark uses the `requires_layout_update=True` metadata flag in the relevant dose-response layout specification.

When this flag is enabled, the benchmark applies `update_compd_layout(...)` before simulating and reading the plate. The conversion performs the following transformation:

1. Preserve the physical position of every well.
2. Preserve `0` as the empty-well code.
3. Preserve the identity of negative-control wells, while updating their code to the new maximum identifier.
4. For each original condition-level identifier, find all corresponding physical wells.
5. Assign a distinct identifier to each physical well in that condition’s replicate group.

In symbolic form, if the original layout has:

```text
C × D experimental IDs
```

and each condition has `R` replicate wells, the conversion changes:

```text
old experimental IDs:       1 ... C × D
old negative-control ID:    C × D + 1
```

into:

```text
new experimental IDs:       1 ... C × D × R
new negative-control ID:    C × D × R + 1
```

For an original condition identifier `i`, numbered from zero, and its replicate occurrence `j`, numbered from zero, the converted identifier is:

```text
new_id = 1 + i + j × C × D
```

This preserves the condition grouping while making every physical observation addressable by the benchmark.

To conclude - if your newly generated layouts produce matrices in the natural condition-level encoding, you have to enable `requires_layout_update` metadata flag.


### Ordering is part of the contract

The conversion assigns replicate-specific IDs by iterating through the matching wells in stable matrix traversal order: row by row, then column by column.

This means that the layout matrix, its encoded values, and the ordering of generated plate content must agree.

A generator that uses a different ordering convention - for example, column-major order, replicate-major ordering that does not match matrix traversal, or a custom compound/dose ordering - may produce a matrix that is formally valid but scientifically misaligned with the simulated data.

Before importing a new layout family, verify all of the following:

- Which compound/concentration condition corresponds to each original positive identifier?
- Whether repeated IDs represent technical replicates.
- The physical traversal order used to enumerate repeated IDs.
- The ordering used by the benchmark to construct `plate_content`.
- The resulting mapping from unique per-well IDs back to compound, dose, and replicate observations.

### How to choose `requires_layout_update`

Use the registry flag as follows:

| Input layout representation | `requires_layout_update` | Reason |
|---|---:|---|
| Every physical experimental well already has a unique positive ID | `False` | The layout is already in benchmark-ready per-well form |
| Technical-replicate wells reuse a condition-level positive ID | `True` | The benchmark must expand repeated condition IDs into unique per-well IDs |
| The encoding is neither compatible condition-level nor benchmark-ready per-well | Do not import yet | The matrix needs an explicit, verified conversion before use |

Do not set `requires_layout_update=True` as a generic workaround. It is specifically a conversion for the known COMPD-style condition-level encoding. Applying it to a layout with an incompatible ID scheme can silently scramble the mapping between layout wells and simulated observations.

### Negative controls are identified by the maximum code

The current dose-response utilities use the largest layout value as the negative-control identifier:

```python
neg_control_id = np.max(layout)
```

Accordingly:

- All negative-control wells must use the same maximum integer code.
- No experimental well may use a value greater than the negative-control code.
- A layout with multiple control types requires a separately verified encoding strategy; it must not assume that all control types can be represented as the current single negative-control maximum value.
- After a `requires_layout_update` conversion, the negative-control code changes from `C × D + 1` to `C × D × R + 1`.

### Recommended validation before benchmarking

For a dose-response layout with `C` compounds, `D` concentrations, and `R` replicates, validate the final matrix after any required conversion:

```python
import numpy as np

layout = np.load("my-layout.npy")

assert layout.ndim == 2
assert np.issubdtype(layout.dtype, np.integer)
assert layout.min() >= 0

negative_control_id = layout.max()

experimental_ids = np.unique(layout[(layout > 0) & (layout < negative_control_id)])

assert len(experimental_ids) == C * D * R
assert np.array_equal(
    experimental_ids,
    np.arange(1, C * D * R + 1),
)

assert negative_control_id == C * D * R + 1
```

This confirms the basic benchmark-ready per-well encoding. It does not by itself prove that the IDs correspond to the intended compound, concentration, and replicate order; that mapping should be checked with a small hand-audited layout before a new generator is used at full scale.

### Exporting layouts to `.npy`

A generator should export the final numeric matrix using NumPy, for example:

```python
import numpy as np

# plate_matrix must be a two-dimensional integer array:
# shape == (rows, columns)
# dtype should be an integer dtype.
np.save("my-layout.npy", plate_matrix.astype(int))
```

Before using the file in the benchmark, verify:

```python
import numpy as np

layout = np.load("my-layout.npy")

assert layout.ndim == 2
assert np.issubdtype(layout.dtype, np.integer)
assert layout.min() >= 0
```

These checks establish only the basic file-format contract. They do not verify that the layout has the correct experimental capacity, control placement, or ordering for a particular benchmark scenario.

### Filename and directory contract

The file must be placed in a directory and named according to the corresponding layout specification in `benchmark_common.py`.

The registry associates each layout family with metadata such as:

- Its displayed name, for example `COMPD`, `PLAID`, or `Random`.
- The directory containing its `.npy` files.
- The filename pattern used to select applicable matrices.
- Its plot/table order.
- The normalisation or error-correction strategy associated with that layout in the relevant pipeline.

The benchmark does not infer the layout family from the contents of the matrix. It discovers files through the configured directory and filename pattern, then writes the registered `display_type` into downstream CSVs.

Therefore, a custom layout generated by any method can be evaluated as an existing layout family if it follows the appropriate registered filename convention. To introduce it as a separately reported family, add one new registry specification in `benchmark_common.py`.

### Layout families currently compared

The default benchmark configuration includes these reported layout families:

| Layout family | Benchmark role | Generator requirement |
|---|---|---|
| `COMPD` | Constraint-optimised layout family | May be generated by COMPD or reproduced by another compatible generator |
| `PLAID` | PLAID/effective-layout comparison family | Must preserve the expected PLAID-compatible encoding and registered file convention |
| `Random` | Randomised comparison baseline | Must preserve the required plate dimensions, capacity, control encoding, and file convention |

These names identify the benchmark groups shown in figures and tables. They do not impose a technical restriction that the matrices must have been generated by a particular implementation.

For example, a newly developed optimiser may generate a PLAID-format-compatible matrix and be evaluated under a new `MyMethod` registry entry. Or, a matrix produced by a new version of COMPD (e.g., 1.34) can be reported as `COMPD134`.

### Effective-layout design principles

Regardless of the generation method, candidate layouts should be validated against the design properties relevant to the experiment:

- Balance controls across rows, columns, plate halves, and plate regions where possible.
- Keep controls of the same type sufficiently separated.
- Spread technical replicates across rows and columns.
- For dose-response experiments, spread concentrations of a compound across the plate rather than concentrating them in one region.
- Apply the intended edge-well policy consistently.
- Avoid creating large, systematic empty regions when unused wells are present.

PLAID provides a useful reference implementation of these constraint-based design ideas, including edge-well choices, concentration placement, replication constraints, control balancing, and control separation. [PLAID documentation](https://github.com/pharmbio/plaid) should be consulted when reproducing or extending its encoding and generation semantics.

### Safe import workflow

Adding a newly generated layout is a scientific/configuration change. Use the following sequence:

1. Generate the candidate layout with any suitable model or tool.
2. Export it as a two-dimensional integer `.npy` matrix with PLAID-compatible well codes.
3. Check its shape, integer type, value range, control count, and experimental capacity.
4. Place it in the registered directory using a filename that matches the relevant pattern.
5. Add or update one layout entry in `benchmark_common.py` if the layout needs a new reported identity.
6. Run the relevant `simulate` stage.
7. Verify the generated CSV contents and filenames.
8. Run `figures`, `metrics` where applicable, and `tables`.
9. Inspect the generated LaTeX for unexpected `% MISSING:` comments.
10. Review the resulting plots and tables before using the new layout in scientific comparisons.

Do not add a second hard-coded layout list inside simulation, plotting, table-writing, or LaTeX-generation code. Layout identity and file-discovery metadata belong in `benchmark_common.py`.



## Editing benchmark layouts registry

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

### Currently enabled disturbance scenarios

The registry can describe more scenarios than are currently included in the published benchmark output. The `publish_dr` and `publish_screening` flags determine which disturbance families are emitted by the corresponding generated supplementary section.

The currently enabled scenarios should be read from `benchmark_disturbances.py`; the following summary describes the default published configuration.

#### Dose-response

The dose-response pipeline currently publishes three disturbance families:

| Registry-level scenario | Description | Published output |
|---|---|---|
| Bowl-shaped plate effects, negative controls unaffected | A bowl-shaped positional effect applied while preserving the intended negative-control reference behaviour | Dose-response figures, tables, and a subsection in `dr_section_auto.tex` |
| Bowl-shaped plate effects, negative controls affected | A bowl-shaped positional effect that also affects negative controls | Dose-response figures, tables, and a subsection in `dr_section_auto.tex` |
| Half-column/right-half plate effects | A positional effect affecting one side of the plate, used as the column/half-plate scenario | Dose-response figures, tables, and a subsection in `dr_section_auto.tex` |

For every published dose-response disturbance, the registry supplies the scenario identifier, error-strength metadata, human-readable caption text, and filename stem/suffix metadata used downstream.

#### Screening

The screening pipeline currently publishes one disturbance family:

| Registry-level scenario | Description | Published output |
|---|---|---|
| Bowl-shaped plate effects | Bowl-shaped positional effects evaluated across the configured screening control arrangements, hit rates, and error strengths | Screening ROC/PR figures, AUC/quality-metric tables, and a subsection in `screening_section_auto.tex` |

The screening registry metadata is intentionally structured to support future published disturbance families. However, adding a registry entry alone does not make a new screening disturbance scientifically valid: the screening simulation pipeline must also know how to apply the disturbance and the expected output artifacts must be generated and verified.

### Checking the active set

When changing publication flags or registry metadata, regenerate the relevant table stage:

```bash
python run_dose_response_benchmark.py --stage tables
python run_screening_benchmark.py --stage tables
```

Then inspect:

```text
detailed-experimental-results-source/tikz-figures/dr_section_auto.tex
detailed-experimental-results-source/tikz-figures/screening_section_auto.tex
```

There should be one generated `\subsection{...}` per published disturbance family and no unexpected `% MISSING:` comments.


## Running the benchmark pipelines

Run all commands from the repository root. The benchmark is split into two independent pipelines:

- `run_dose_response_benchmark.py` for dose-response simulations and artifacts.
- `run_screening_benchmark.py` for high-throughput screening simulations and artifacts.

Both drivers are stage-based. This makes it possible to regenerate only the outputs affected by a change instead of re-running the full simulation benchmark.

### Dose-response pipeline

```bash
python run_dose_response_benchmark.py --stage simulate
python run_dose_response_benchmark.py --stage figures
python run_dose_response_benchmark.py --stage tables
python run_dose_response_benchmark.py --stage curves
```

The stages have the following responsibilities:

| Stage | Purpose | Main outputs |
|---|---|---|
| `simulate` | Runs the dose-response plate simulations, disturbance/error correction, normalisation, and LL.4 curve fitting | CSV files in `generated-data/dose-response/` |
| `figures` | Reads existing CSVs and creates the benchmark figure PNGs | PNG files in `detailed-experimental-results-source/figures/` |
| `tables` | Reads existing CSVs and writes LaTeX table fragments and the generated dose-response supplementary fragment | `.tex` files in `detailed-experimental-results-source/tables/` and `tikz-figures/dr_section_auto.tex` |
| `curves` | Regenerates selected illustrative fitted-curve figures used by the paper | PNG files in `detailed-experimental-results-source/figures/` |
| `all` | Runs the complete dose-response workflow | All of the above |

For a normal documentation/artifact refresh after modifying caption templates, table formatting, registry labels, or LaTeX-generation logic, run only:

```bash
python run_dose_response_benchmark.py --stage tables
```

For a plotting-only change, when the simulation CSVs already exist and remain valid, run:

```bash
python run_dose_response_benchmark.py --stage figures
```

Use `--stage simulate` only when the simulation inputs themselves have changed, for example a layout matrix, a normalisation implementation, or an intentionally changed disturbance definition. Re-running simulation overwrites or regenerates scientific data artifacts and should therefore be treated as a deliberate change.

### Screening pipeline

```bash
python run_screening_benchmark.py --stage simulate
python run_screening_benchmark.py --stage figures
python run_screening_benchmark.py --stage metrics
python run_screening_benchmark.py --stage tables
```

The screening stages are:

| Stage | Purpose | Main outputs |
|---|---|---|
| `simulate` | Runs screening plate simulations, applies the configured disturbance and normalisation, and writes the data needed for hit-identification and ROC/PR analysis | CSV files in `generated-data/screening/` |
| `figures` | Reads existing screening CSVs and creates ROC/PR and other screening figure PNGs | PNG files in `detailed-experimental-results-source/figures/` |
| `metrics` | Generates screening quality-assessment metrics, including the configured Z-factor/SSMD-style metric artifacts and their figures/tables | Screening metric CSVs, PNGs, and/or LaTeX artifacts in the configured generated-data, figures, and tables directories |
| `tables` | Writes screening LaTeX table fragments and the generated screening supplementary fragment | `.tex` files in `detailed-experimental-results-source/tables/` and `tikz-figures/screening_section_auto.tex` |
| `all` | Runs the complete screening workflow | All of the above |

For the common case of refreshing all presentation artifacts from already-generated simulation data:

```bash
python run_screening_benchmark.py --stage figures
python run_screening_benchmark.py --stage metrics
python run_screening_benchmark.py --stage tables
```

For a change limited to screening table formatting, captions, registry labels, or generated supplementary-section wiring, run only:

```bash
python run_screening_benchmark.py --stage tables
```

Use `--stage simulate` only when simulation inputs themselves have intentionally changed, for example the layout matrix, normalisation implementation, control configuration, or disturbance definition.

### Recommended workflows

**Regenerate only supplementary LaTeX/table artifacts**

```bash
python run_dose_response_benchmark.py --stage tables
python run_screening_benchmark.py --stage tables
```

**Regenerate figures and tables from unchanged simulation CSVs**

```bash
python run_dose_response_benchmark.py --stage figures
python run_dose_response_benchmark.py --stage tables

python run_screening_benchmark.py --stage figures
python run_screening_benchmark.py --stage tables
```

**Run a complete reproducibility pass**

```bash
python run_dose_response_benchmark.py --stage all
python run_screening_benchmark.py --stage all
```

### Generated LaTeX is part of `tables`

The `tables` stage writes both ordinary LaTeX table fragments and the generated supplementary section files:

- `detailed-experimental-results-source/tikz-figures/dr_section_auto.tex`
- `detailed-experimental-results-source/tikz-figures/screening_section_auto.tex`

These files are generated from benchmark configuration, registry metadata, filename conventions, and the artifacts that exist on disk. Do not edit them manually.

The supplement imports the generated sections with:

```tex
\input{tikz-figures/dr_section_auto}
\input{tikz-figures/screening_section_auto}
```

Consequently, adding a generated figure/table to the supplement normally requires changing the relevant Python configuration or generator logic and re-running `--stage tables`; it should not require editing the supplement’s LaTeX wrappers by hand.

If an expected figure or table does not exist, the generated section contains a `% MISSING: ...` comment rather than failing. Inspect these comments before compiling the supplementary material.


## Repository data flow

```text
Layout matrices (.npy) + benchmark configuration
                    |
                    v
        simulate stage -> generated-data/
                    |
                    +-------------------+
                    |                   |
                    v                   v
          figures / metrics          tables
                    |                   |
                    v                   v
              figures/*.png      tables/*.tex
                    \                   /
                     \                 /
                      v               v
          tikz-figures/*_section_auto.tex
                      |
                      v
             0_supplement.tex / paper LaTeX
```

The benchmark scripts own generated CSVs, PNGs, LaTeX tables, and `*_section_auto.tex` files. Manual/semantic TikZ layouts remain separate and are maintained by hand.


## Reproducibility policy

The repository distinguishes between two types of changes.

### Presentation/configuration changes

Examples include:

- Caption wording.
- Registry display labels.
- LaTeX table formatting.
- Generated-section layout.
- Plot styling that does not alter the underlying calculations.

These changes normally require only the affected `figures`, `metrics`, or `tables` stage.

### Scientific/data-generating changes

Examples include:

- A layout matrix.
- Plate encoding or `requires_layout_update` behaviour.
- A normalisation method.
- A disturbance function or error strength.
- Simulation parameters, such as dose count, replicate count, control configuration, or hit rate.
- A metric calculation.

These changes require regenerating downstream artifacts, starting with `simulate`, then the applicable `figures`, `metrics`, and `tables` stages.

Do not combine outputs generated under different scientific configurations in the same supplement or manuscript build.
