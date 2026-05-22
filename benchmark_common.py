from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional
import os


DILUTION_BY_CONCENTRATIONS = {
    4: 15,
    6: 18,
    8: 8,
    12: 4,
}


def dilution_for(concentrations: int) -> int:
    try:
        return DILUTION_BY_CONCENTRATIONS[concentrations]
    except KeyError as exc:
        raise ValueError(f"Unsupported concentrations value: {concentrations}") from exc


def fig_dir_str(path: Path) -> str:
    return str(path) + os.sep


def _default_error_correction() -> Callable:
    """Return the default plate-normalisation function.

    The import is deferred to this function so that benchmark_common.py does not
    import libraries.normalization at module load time.  This breaks the circular
    import chain:

        normalization → utilities → benchmark_common → normalization  ✗

    Callers that need the default should use this helper instead of storing the
    function object directly in the dataclass default.
    """
    import libraries.normalization as nrm  # noqa: PLC0415
    return nrm.normalize_plate_lowess_2d


@dataclass(frozen=True)
class LayoutSpec:
    key: str
    display_type: str
    layout_dir: str
    regex_template: str
    # None means "use _default_error_correction() at call time".
    # Storing an explicit callable overrides the default.
    error_correction: Optional[Callable] = None
    color: str = ""           # plot colour for this layout (ROC/PR/scatter figures)
    residuals_color: str = "" # residual plot colour for this layout (shades of purple)
    plot_order: int = 0       # used for sorted() calls in LaTeX table generation
    residuals_plot_order: int = 0 # used for residual plots
    requires_layout_update: bool = False  # True for layouts (e.g. COMPD) that need
                                          # update_compd_layout() applied after loading
    curve_example_file: Optional[str] = None
    curve_example_compounds: Optional[int] = None
    curve_example_concentrations: Optional[int] = None
    curve_example_replicates: Optional[int] = None
    control_example_file: Optional[str] = None
    control_figure_output: Optional[str] = None

    def _resolved_error_correction(self) -> Callable:
        """Return the error-correction callable, resolving the default lazily."""
        if self.error_correction is not None:
            return self.error_correction
        return _default_error_correction()

    def as_dict(self, **fmt: Any) -> Dict[str, Any]:
        return {
            # short key used in CSV layout column etc.
            "type": self.display_type,
            # explicit display label expected by full_dose_response_evaluation
            "display_type": self.display_type,
            "dir": self.layout_dir,
            "regex": self.regex_template.format(**fmt),
            "error_correction": self._resolved_error_correction(),
            "requires_layout_update": self.requires_layout_update,
        }


DOSE_RESPONSE_LAYOUT_SPECS: List[LayoutSpec] = [
    LayoutSpec(
        key="compd",
        display_type="COMPD",
        layout_dir="layouts/compounds_COMPD_layouts/",
        regex_template=r"plate_layout_(.*){compounds}-{concentrations}-{replicates}_(0*)(.+?).npy",
        requires_layout_update=True,
        color="#1b9e77",  # green
        residuals_color="#37185d",
        plot_order=0,
        residuals_plot_order=2,
        curve_example_file="plate_layout_40-12-8-3_01.npy",
        curve_example_compounds=12,
        curve_example_concentrations=8,
        curve_example_replicates=3,
    ),
    LayoutSpec(
        key="plaid",
        display_type="PLAID",
        layout_dir="layouts/compounds_PLAID_layouts/",
        regex_template=r"plate_layout_(.*){compounds}-{concentrations}-{replicates}_(0*)(.+?).npy",
        color="#d95f02",  # orange
        residuals_color="#765591",
        plot_order=1,
        residuals_plot_order=1,
        curve_example_file="plate_layout_20-12-8-3_01.npy",
        curve_example_compounds=12,
        curve_example_concentrations=8,
        curve_example_replicates=3,
    ),
    LayoutSpec(
        key="random",
        display_type="Random",
        layout_dir="layouts/compounds_manual_layouts/",
        regex_template=r"plate_layout_rand_(.+?).npy",
        color="#7570b3",  # purple
        residuals_color="#b7a2d8",
        plot_order=2,
        residuals_plot_order=0,
        curve_example_file="plate_layout_rand_02.npy",
        curve_example_compounds=12,
        curve_example_concentrations=8,
        curve_example_replicates=3,
    ),
]


SCREENING_LAYOUT_SPECS: List[LayoutSpec] = [
    LayoutSpec(
        key="random",
        display_type="Random",
        layout_dir="layouts/screening_RANDM_layouts/",
        regex_template=r"plate_layout_rand_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
        color="#59296e",
        plot_order=0,
        control_example_file="plate_layout_rand_10-10_02.npy",
        control_figure_output="figures/plate_random-controls-rows-error.png",
    ),
    LayoutSpec(
        key="plaid",
        display_type="PLAID",
        layout_dir="layouts/screening_PLAID_layouts/",
        regex_template=r"plate_layout_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
        color="#cc0253",
        plot_order=1,
        control_example_file="plate_layout_10-10_01.npy",
        control_figure_output="figures/plate_plaid-controls-rows-error.png",
    ),
    LayoutSpec(
        key="compd",
        display_type="COMPD",
        layout_dir="layouts/screening_COMPD_layouts/",
        regex_template=r"plate_layout_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
        color="#e68302",
        plot_order=2,
        control_example_file="plate_layout_10-10_01.npy",
        control_figure_output="figures/plate_compd-controls-rows-error.png",
    ),
]

SCREENING_LAYOUT_ORDER = [spec.display_type for spec in SCREENING_LAYOUT_SPECS]
SCREENING_LAYOUT_BOX_PAIRS = [
    (SCREENING_LAYOUT_ORDER[i], SCREENING_LAYOUT_ORDER[j])
    for i in range(len(SCREENING_LAYOUT_ORDER))
    for j in range(i + 1, len(SCREENING_LAYOUT_ORDER))
]


def screening_plate_types(neg_controls: int, pos_controls: int):
    return [
        {
            "type": spec.key,
            "display_type": spec.display_type,
            "dir": spec.layout_dir,
            "regex": spec.regex_template.format(
                neg_controls=neg_controls,
                pos_controls=pos_controls,
            ),
            "error_correction": spec._resolved_error_correction(),
        }
        for spec in SCREENING_LAYOUT_SPECS
    ]


def screening_metrics_plate_types(neg_controls: int, pos_controls: int):
    return [
        {
            "type": spec.key,
            "display_type": spec.display_type,
            "dir": spec.layout_dir,
            "regex": spec.regex_template.format(
                neg_controls=neg_controls,
                pos_controls=pos_controls,
            ),
        }
        for spec in SCREENING_LAYOUT_SPECS
    ]


def screening_control_figure_cases():
    return [
        (os.path.join(spec.layout_dir, spec.control_example_file), spec.control_figure_output)
        for spec in SCREENING_LAYOUT_SPECS
        if spec.control_example_file is not None and spec.control_figure_output is not None
    ]

DOSE_RESPONSE_LAYOUT_ORDER = [
    spec.display_type for spec in sorted(DOSE_RESPONSE_LAYOUT_SPECS, key=lambda s: s.residuals_plot_order)
]
DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER = [
    spec.display_type
    for spec in sorted(DOSE_RESPONSE_LAYOUT_SPECS, key=lambda s: s.residuals_plot_order)
]
DOSE_RESPONSE_LAYOUT_BOX_PAIRS = [
    (DOSE_RESPONSE_LAYOUT_ORDER[i], DOSE_RESPONSE_LAYOUT_ORDER[j])
    for i in range(len(DOSE_RESPONSE_LAYOUT_ORDER))
    for j in range(i + 1, len(DOSE_RESPONSE_LAYOUT_ORDER))
]
DOSE_RESPONSE_LAYOUT_BOX_PAIRS_BY_REPLICATE = [
    ((rep, DOSE_RESPONSE_LAYOUT_ORDER[i]), (rep, DOSE_RESPONSE_LAYOUT_ORDER[j]))
    for rep in (1, 2, 3)
    for i in range(len(DOSE_RESPONSE_LAYOUT_ORDER))
    for j in range(i + 1, len(DOSE_RESPONSE_LAYOUT_ORDER))
]


def dose_response_plate_types(compounds: int, concentrations: int, replicates: int):
    return [
        spec.as_dict(
            compounds=compounds,
            concentrations=concentrations,
            replicates=replicates,
        )
        for spec in DOSE_RESPONSE_LAYOUT_SPECS
    ]


def dose_response_curve_examples():
    return [
        (
            spec.display_type,
            spec.layout_dir,
            spec.curve_example_file,
            spec.curve_example_compounds,
            spec.curve_example_concentrations,
            spec.curve_example_replicates,
        )
        for spec in DOSE_RESPONSE_LAYOUT_SPECS
        if spec.curve_example_file is not None
    ]

DOSE_RESPONSE_FIGURE_CASES = [(6, 18), (8, 8), (12, 4)]
BOWL_ERROR_LEVELS = (0.055, 0.085)
RIGHT_HALF_ERROR_LEVELS = (0.2, 0.4)

def classify_layout_by_key(key: str, specs) -> str:
    """Return display_type for a layout key. Raises ValueError for unknown keys."""
    mapping = {s.key: s.display_type for s in specs}
    if key not in mapping:
        raise ValueError(f"Unknown layout key {key!r}. Known: {sorted(mapping)}")
    return mapping[key]


def validate_layout_registry_consistency():
    """Assert that dose-response and screening registries cover the same keys."""
    expected = {"compd", "plaid", "random"}
    dr_keys = {s.key for s in DOSE_RESPONSE_LAYOUT_SPECS}
    sr_keys = {s.key for s in SCREENING_LAYOUT_SPECS}
    assert dr_keys == expected, f"Dose-response keys mismatch: {dr_keys}"
    assert sr_keys == expected, f"Screening keys mismatch: {sr_keys}"
    print("Layout registries are consistent.")
