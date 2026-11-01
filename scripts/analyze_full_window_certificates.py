"""Pool the complete-window class-margin certificate counts."""

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.certificates import certificate_counts
from representative_context.runtime import read_config, write_json


def analyze(document):
    rows = document["windows"]
    if len({row["validation_index"] for row in rows}) != len(rows):
        raise ValueError("Window identities must be unique.")
    for row in rows:
        if not 0 <= row["certified_pixels"] <= row["valid_pixels"]:
            raise ValueError("Certificate counts must lie within the valid-pixel counts.")
    certified = sum(row["certified_pixels"] for row in rows)
    valid = sum(row["valid_pixels"] for row in rows)
    if valid <= 0:
        raise ValueError("At least one valid pixel is required.")
    return dict(windows=len(rows), certified_pixels=certified, valid_pixels=valid,
                certified_percent=100 * certified / valid, predicate=document["predicate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/03_theory/full_window_certificates.json")
    parser.add_argument("--fields-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = read_config(args.input)
    if args.fields_dir is not None:
        for row in document["windows"]:
            if "fields" not in row:
                raise ValueError("Use a newly collected manifest with saved per-pixel fields.")
            with np.load(args.fields_dir / row["fields"], allow_pickle=False) as fields:
                counts = certificate_counts(**{key: fields[key] for key in
                    ("centered_error_energy", "minimum_dense_margin", "valid_mask")})
            for key in ("certified_pixels", "valid_pixels"):
                if counts[key] != row[key]:
                    raise ValueError("Saved certificate counts disagree with the per-pixel predicate.")
    result = analyze(document)
    result["validation"] = "per_pixel_fields_checked" if args.fields_dir is not None else "archived_count_totals_checked"
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
