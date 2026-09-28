"""
Build pipeline/reference_ranges.json: min/max of every stat-vector
coefficient across the 2019-2025 leave-one-season-out fits (the folds behind
the README tables). Retraining validates new coefficients against these.

Run: python -m pipeline.reference_ranges   (rerun only when a spec changes)
"""
import json
from datetime import datetime, timezone

import polars as pl

from clients.nflreadpy.pbp_data import load_pbp_data
from clients.nflreadpy.player_data import load_player_stats
from evaluation.backtest import FULL_SEASONS, LOAD_SEASONS, PBP_COLUMNS
from pipeline.artifacts import REFERENCE_RANGES_PATH, spec_hash
from projections.expected_points.features.epa_allowed import calculate_epa_allowed
from projections.expected_points.stat_vector.core import fit_stat_vector
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS


def build() -> dict:
    stats = load_player_stats(LOAD_SEASONS)
    pbp = pl.concat([load_pbp_data(s).select(PBP_COLUMNS) for s in LOAD_SEASONS], how="vertical_relaxed")
    epa = calculate_epa_allowed(pbp)

    ranges = {}
    for pos, spec in STAT_VECTOR_SPECS.items():
        per_term = {}
        for test in FULL_SEASONS:
            train = [s for s in FULL_SEASONS if s != test]
            model = fit_stat_vector(stats, epa, spec, train)
            for target, fit in model.fits.items():
                for term, value in zip(["intercept"] + fit.features, fit.coef):
                    per_term.setdefault(target, {}).setdefault(term, []).append(float(value))
        ranges[pos] = {
            target: {term: {"min": min(v), "max": max(v), "n_folds": len(v)} for term, v in terms.items()}
            for target, terms in per_term.items()
        }

    return {
        "description": "Stat-vector coefficient ranges across 2019-2025 leave-one-season-out fits",
        "folds": FULL_SEASONS,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_hashes": {pos: spec_hash(spec) for pos, spec in STAT_VECTOR_SPECS.items()},
        "ranges": ranges,
    }


if __name__ == "__main__":
    out = build()
    REFERENCE_RANGES_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {REFERENCE_RANGES_PATH}")
