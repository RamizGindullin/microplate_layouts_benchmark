from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict, Any
import os

import libraries.normalization as nrm


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


@dataclass(frozen=True)
class LayoutSpec:
    key: str
    display_type: str
    layout_dir: str
    regex_template: str
    error_correction: Callable = nrm.normalize_plate_lowess_2d
    color: str = ""  # plot colour for this layout (used in ROC/PR/scatter figures)

    def as_dict(self, **fmt: Any) -> Dict[str, Any]:
        return {
            "type": self.display_type,
            "dir": self.layout_dir,
            "regex": self.regex_template.format(**fmt),
            "error_correction": self.error_correction,
        }


DOSE_RESPONSE_LAYOUT_SPECS: List[LayoutSpec] = [
    LayoutSpec(
        key="compd",
        display_type="COMPD",
        layout_dir="layouts/compounds_COMPD_layouts/",
        regex_template=r"plate_layout_(.*){compounds}-{concentrations}-{replicates}_(0*)(.+?).npy",
    ),
    LayoutSpec(
        key="plaid",
        display_type="PLAID",
        layout_dir="layouts/compounds_PLAID_layouts/",
        regex_template=r"plate_layout_(.*){compounds}-{concentrations}-{replicates}_(0*)(.+?).npy",
    ),
    LayoutSpec(
        key="random",
        display_type="Random",
        layout_dir="layouts/compounds_manual_layouts/",
        regex_template=r"plate_layout_rand_(.+?).npy",
    ),
]


SCREENING_LAYOUT_SPECS: List[LayoutSpec] = [
    LayoutSpec(
        key="random",
        display_type="Random",
        layout_dir="layouts/screening_RANDM_layouts/",
        regex_template=r"plate_layout_rand_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
        color="#59296e",
    ),
    LayoutSpec(
        key="plaid",
        display_type="PLAID",
        layout_dir="layouts/screening_PLAID_layouts/",
        regex_template=r"plate_layout_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
        color="#cc0253",
    ),
    LayoutSpec(
        key="compd",
        display_type="COMPD",
        layout_dir="layouts/screening_COMPD_layouts/",
        regex_template=r"plate_layout_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
        color="#e68302",
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
            "type": spec.display_type,
            "display_type": spec.display_type,
            "dir": spec.layout_dir,
            "regex": spec.regex_template.format(
                neg_controls=neg_controls,
                pos_controls=pos_controls,
            ),
            "error_correction": spec.error_correction,
        }
        for spec in SCREENING_LAYOUT_SPECS
    ]


def screening_metrics_plate_types(neg_controls: int, pos_controls: int):
    return [
        {
            "type": spec.display_type,
            "display_type": spec.display_type,
            "dir": spec.layout_dir,
            "regex": spec.regex_template.format(
                neg_controls=neg_controls,
                pos_controls=pos_controls,
            ),
        }
        for spec in SCREENING_LAYOUT_SPECS
    ]

DOSE_RESPONSE_LAYOUT_ORDER = [spec.display_type for spec in DOSE_RESPONSE_LAYOUT_SPECS]
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