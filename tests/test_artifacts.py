"""Artifact safety and retrain determinism."""
import copy
import json
from dataclasses import replace

import pytest

from pipeline.artifacts import (
    ArtifactSpecMismatch, ArtifactValidationError, artifact_to_model, check_compatible, read_all, read_artifact,
    write_all,
)
from pipeline.train import train_models
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.parent.rglob("*")) if p.is_file()}


def test_spec_mismatch_raises(trained):
    directory, _ = trained
    art = read_artifact("RB", directory)
    check_compatible(art)  # the real one is fine

    bad_window = replace(STAT_VECTOR_SPECS["RB"], window=STAT_VECTOR_SPECS["RB"].window + 1)
    with pytest.raises(ArtifactSpecMismatch, match="window"):
        artifact_to_model(art, bad_window)

    swapped = copy.deepcopy(art)
    feats = swapped["sub_models"]["ypc"]["features"]
    feats[0], feats[1] = feats[1], feats[0]
    with pytest.raises(ArtifactSpecMismatch, match="features"):
        check_compatible(swapped)

    renamed = copy.deepcopy(art)
    renamed["sub_models"]["ypc"]["features"][2] = "epa_allowed_pass"
    with pytest.raises(ArtifactSpecMismatch):
        artifact_to_model(renamed)


def test_failed_validation_leaves_artifacts_intact(trained):
    directory, result = trained
    before = _snapshot(directory)
    bad = copy.deepcopy(result["artifacts"])
    bad["QB"]["sub_models"]["attempts"]["coefficients"][0] = float("nan")
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        write_all(bad, "should never be written", directory)
    assert _snapshot(directory) == before
    assert not any(p.name.startswith(".tmp") for p in directory.parent.iterdir())

    too_small = copy.deepcopy(result["artifacts"])
    too_small["TE"]["sub_models"]["targets"]["n"] = 10
    with pytest.raises(ArtifactValidationError, match="floor"):
        write_all(too_small, "nope", directory)
    assert _snapshot(directory) == before


def test_retrain_is_deterministic(ctx, trained):
    directory, _ = trained
    _, first, _, _ = train_models(ctx, 2022)
    _, second, _, _ = train_models(ctx, 2022)
    saved = read_all(directory)
    for pos in STAT_VECTOR_SPECS:
        a, b = json.dumps(first[pos], sort_keys=True), json.dumps(second[pos], sort_keys=True)
        assert a == b, f"{pos}: two consecutive fits differ"
        on_disk = {k: v for k, v in saved[pos].items() if k != "fit_timestamp"}
        assert json.dumps(on_disk, sort_keys=True) == json.dumps(json.loads(a), sort_keys=True), \
            f"{pos}: saved artifact differs from a fresh fit (beyond the timestamp)"
