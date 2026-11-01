"""Check newly collected latency and speedup summaries against original calls."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from representative_context.analysis.timing_records import call_statistics, round_medians, validate_report
from representative_context.runtime import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = read_config(args.input)
    if "strata" in document:
        result = validate_report(document)
    elif "timing_ms" in document:
        result = dict(**call_statistics(document["timing_ms"]),
                      round_median_ms=round_medians(document))
    else:
        raise ValueError("This entry requires original call records, not a summary-only result.")
    write_json(args.output, dict(validation="raw_calls_checked", **result))
    print(args.output)


if __name__ == "__main__":
    main()
