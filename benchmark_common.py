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
        display_type="RANDOM",
        layout_dir="layouts/compounds_manual_layouts/",
        regex_template=r"plate_layout_rand_(.+?).npy",
    ),
]


SCREENING_LAYOUT_SPECS: List[LayoutSpec] = [
    LayoutSpec(
        key="random",
        display_type="random",
        layout_dir="layouts/screening_RANDM_layouts/",
        regex_template=r"plate_layout_rand_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
    ),
    LayoutSpec(
        key="plaid",
        display_type="plaid",
        layout_dir="layouts/screening_PLAID_layouts/",
        regex_template=r"plate_layout_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
    ),
    LayoutSpec(
        key="compd",
        display_type="compd",
        layout_dir="layouts/screening_COMPD_layouts/",
        regex_template=r"plate_layout_{neg_controls}-{pos_controls}_(0*)(.+?).npy",
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

DOSE_RESPONSE_LAYOUT_DISPLAY_TO_LEGACY = {
    "COMPD": "Border",
    "PLAID": "Random",
    "RANDOM": "Effective",
}
DOSE_RESPONSE_LAYOUT_LEGACY_ORDER = [
    DOSE_RESPONSE_LAYOUT_DISPLAY_TO_LEGACY[name]
    for name in DOSE_RESPONSE_LAYOUT_ORDER
]
DOSE_RESPONSE_LAYOUT_LEGACY_BOX_PAIRS = [
    (DOSE_RESPONSE_LAYOUT_LEGACY_ORDER[i], DOSE_RESPONSE_LAYOUT_LEGACY_ORDER[j])
    for i in range(len(DOSE_RESPONSE_LAYOUT_LEGACY_ORDER))
    for j in range(i + 1, len(DOSE_RESPONSE_LAYOUT_LEGACY_ORDER))
]
DOSE_RESPONSE_LAYOUT_LEGACY_BOX_PAIRS_BY_REPLICATE = [
    ((rep, DOSE_RESPONSE_LAYOUT_LEGACY_ORDER[i]), (rep, DOSE_RESPONSE_LAYOUT_LEGACY_ORDER[j]))
    for rep in (1, 2, 3)
    for i in range(len(DOSE_RESPONSE_LAYOUT_LEGACY_ORDER))
    for j in range(i + 1, len(DOSE_RESPONSE_LAYOUT_LEGACY_ORDER))
]


def dose_response_legacy_label(display_name: str) -> str:
    return DOSE_RESPONSE_LAYOUT_DISPLAY_TO_LEGACY[display_name]


def dose_response_legacy_labels(display_names):
    return [dose_response_legacy_label(name) for name in display_names]


def dose_response_legacy_box_pairs(display_names=None):
    labels = DOSE_RESPONSE_LAYOUT_LEGACY_ORDER if display_names is None else dose_response_legacy_labels(display_names)
    return [
        (labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]


def dose_response_legacy_box_pairs_by_replicate(display_names=None, replicates=(1, 2, 3)):
    labels = DOSE_RESPONSE_LAYOUT_LEGACY_ORDER if display_names is None else dose_response_legacy_labels(display_names)
    return [
        ((rep, labels[i]), (rep, labels[j]))
        for rep in replicates
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
