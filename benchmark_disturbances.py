"""Central registry of disturbance families used by the COMPD/PLAID benchmark.

- This module owns:
    • stable machine keys  (DisturbanceScenario.key)
    • human-readable LaTeX labels  (emph_name, long_label)
    • linkage between disturbance families and pipeline IDs
      (dr_id_text, screening_type)
    • per-pipeline publish flags  (publish_dr, publish_screening)
    • filename-safe suffixes used in LaTeX table stems
      (dr_file_suffix, dr_stem_label)

- benchmark_common.py owns:
    • layout lists (DOSE_RESPONSE_LAYOUT_SPECS, SCREENING_LAYOUT_SPECS)
    • structural scenario groupings (BOWL_ERROR_LEVELS, RIGHT_HALF_ERROR_LEVELS,
      DOSE_RESPONSE_FIGURE_CASES, SCREENING_ROC_PR_CASES, etc.)

- The benchmark scripts (run_dose_response_benchmark.py,
  run_screening_benchmark.py) own:
    • how registry entries and layout lists are woven into PNG/LaTeX filenames
      (via lambdas such as IC50_DMAX_R2_SCENARIO_GROUPS)
    • stage orchestration and CLI

Adding a new disturbance family: add one DisturbanceScenario entry here,
set publish_dr / publish_screening as needed, then re-run the relevant stages.
No other file needs to be edited for the disturbance metadata itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Disturbance function map
# ---------------------------------------------------------------------------
# Maps screening_type / dr function-key strings to the actual callables.
# Import is deferred inside the helper functions below so that this module
# remains importable even when libraries/ is not on the path (e.g. in tests).
#
# NOTE: do NOT call these directly from LaTeX generators — the functions are
# only needed by simulation stages. LaTeX generators work entirely from
# existing files on disk.
# ---------------------------------------------------------------------------

def _disturbance_function_for_screening_type(screening_type: str) -> Callable:
    """Return the error-function callable for a given screening_type string."""
    import libraries.disturbances as dt  # noqa: PLC0415
    _MAP: Dict[str, Callable] = {
        "bowl-nl": dt.add_bowlshaped_errors_nl,
    }
    try:
        return _MAP[screening_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown screening_type {screening_type!r}. "
            f"Known: {sorted(_MAP)}"
        ) from exc


def _disturbance_function_for_dr_id(dr_id_text: str) -> Callable:
    """Return the error-function callable for a given dr_id_text string."""
    import libraries.disturbances as dt  # noqa: PLC0415
    _MAP: Dict[str, Callable] = {
        "curve_info-new-reg":                 dt.add_bowlshaped_errors_nl,
        "bowl-neg-control-new-reg":           dt.add_bowlshaped_errors_nl,
        "right-half-neg-control-log-new-reg": dt.add_errors_to_right_columns_half,
    }
    try:
        return _MAP[dr_id_text]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dr_id_text {dr_id_text!r}. "
            f"Known: {sorted(_MAP)}"
        ) from exc


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DisturbanceScenario:
    """
    Metadata for one disturbance family.

    Fields
    ------
    key             Stable machine identifier, e.g. "bowl_nl_neg_unaffected".
                    Used as a dict key and in generated filenames.
    emph_name       Word to wrap in \\emph{} in LaTeX captions,
                    e.g. "bowl-shaped".
    long_label      Short caption label, e.g. "bowl (negatives unaffected)".
                    Used in overview table row headers and LaTeX captions.
    dr_id_text      The id_text suffix used in dose-response CSV filenames.
                    None if this disturbance has no DR scenario.
    screening_type  The "type" string written into screening CSV rows.
                    None if this disturbance has no screening scenario.
    publish_dr      Whether to include in the DR section generator.
    publish_screening
                    Whether to include in the screening section generator.
    dr_file_suffix  Short suffix used in DR overview-table LaTeX filenames,
                    e.g. "bowl" -> dr-overview-rel-ic50-bowl.tex.
                    Must match existing \\input{} calls in 0_supplement.tex.
                    None if publish_dr is False.
    dr_stem_label   Label used in per-scenario DR table filename stems,
                    e.g. "bowl-neg-controls" ->
                    dr-residuals-mean-std-8doses-...-bowl-neg-controls-0.055.tex.
                    None if publish_dr is False.
    """
    key: str
    emph_name: str
    long_label: str
    dr_id_text: Optional[str]
    screening_type: Optional[str]
    publish_dr: bool
    publish_screening: bool
    dr_file_suffix: Optional[str] = None
    dr_stem_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DISTURBANCES: List[DisturbanceScenario] = [
    DisturbanceScenario(
        key="bowl_nl_neg_unaffected",
        emph_name="bowl-shaped",
        long_label="bowl (negatives unaffected)",
        dr_id_text="curve_info-new-reg",
        screening_type="bowl-nl",
        publish_dr=True,
        publish_screening=True,
        dr_file_suffix="bowl",
        dr_stem_label="bowl",
    ),
    DisturbanceScenario(
        key="bowl_nl_neg_affected",
        emph_name="bowl-shaped",
        long_label="bowl (negatives affected)",
        dr_id_text="bowl-neg-control-new-reg",
        screening_type="bowl-nl",
        publish_dr=True,
        publish_screening=False,  # screening uses bowl-nl only once (neg-unaffected)
        dr_file_suffix="bowl-neg",
        dr_stem_label="bowl-neg-controls",
    ),
    DisturbanceScenario(
        key="half_column_neg_affected",
        emph_name="half-column",
        long_label="column (half-plate)",
        dr_id_text="right-half-neg-control-log-new-reg",
        screening_type=None,
        publish_dr=True,
        publish_screening=False,
        dr_file_suffix="column",
        dr_stem_label="half-columns-neg-controls",
    ),
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def dr_scenarios() -> List[DisturbanceScenario]:
    """Return all disturbances that should appear in the DR pipeline."""
    return [d for d in DISTURBANCES if d.publish_dr and d.dr_id_text is not None]


def screening_disturbances() -> List[DisturbanceScenario]:
    """Return all disturbances that should appear in the screening pipeline."""
    return [d for d in DISTURBANCES if d.publish_screening and d.screening_type is not None]


# ---------------------------------------------------------------------------
# Derived lookup tables (built once at import time)
# ---------------------------------------------------------------------------
# These are the single authoritative sources for label and suffix lookups.
# Scripts import these dicts directly rather than building their own.

# dr_id_text → long human label ("bowl (negatives unaffected)", etc.)
DR_LABEL_BY_ID: Dict[str, str] = {
    d.dr_id_text: d.long_label
    for d in DISTURBANCES
    if d.dr_id_text is not None
}

# dr_id_text → short overview-table file suffix ("bowl", "bowl-neg", "column")
# Must match existing \input{} stems in 0_supplement.tex.
DR_FILE_SUFFIX_BY_ID: Dict[str, str] = {
    d.dr_id_text: d.dr_file_suffix
    for d in DISTURBANCES
    if d.dr_id_text is not None and d.dr_file_suffix is not None
}

# dr_id_text → per-scenario table stem label ("bowl", "bowl-neg-controls", etc.)
DR_STEM_LABEL_BY_ID: Dict[str, str] = {
    d.dr_id_text: d.dr_stem_label
    for d in DISTURBANCES
    if d.dr_id_text is not None and d.dr_stem_label is not None
}