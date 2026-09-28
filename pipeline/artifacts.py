"""
Stat-vector model artifacts: JSON on disk, one file per position.

artifacts/stat_vector/<POS>.json holds everything needed to rebuild the
fitted StatVectorModel without refitting, plus provenance. The loader checks
the artifact against the current in-code StatVectorSpec (features, order,
denominators, derivations, window, weighting) and refuses to apply
coefficients on any mismatch.
"""
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from projections.expected_points.stat_vector.core import OLSFit, StatVectorModel, StatVectorSpec
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
STAT_VECTOR_DIR = ARTIFACTS_ROOT / "stat_vector"
REFERENCE_RANGES_PATH = Path(__file__).resolve().parent / "reference_ranges.json"

# Row floors per sub-model fit. Well under what a normal 4+ season window
# produces (smallest: TE rushing rates, a few hundred rows); a count below
# these means the data load or cut went wrong.
MIN_ROWS = {"volume": 1000, "rate": 50}


class ArtifactSpecMismatch(ValueError):
    """An artifact was fit for a different spec than the one in code."""


class ArtifactValidationError(ValueError):
    """A freshly trained artifact set failed validation; nothing was written."""


def _round(x):
    # 12 significant digits: far below any meaningful precision, and it keeps
    # repeated retrains byte-identical when multi-threaded aggregation changes
    # the last bit of a float.
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return x
    return float(f"{x:.12g}")


def spec_config(spec: StatVectorSpec, weight_rates: bool) -> dict:
    return {
        "position": spec.position,
        "window": spec.window,
        "weight_rates": weight_rates,
        "models": [[t, list(feats), den] for t, (feats, den) in spec.models.items()],
        "derivations": [list(d) for d in spec.derivations],
    }


def spec_hash(spec: StatVectorSpec, weight_rates: bool = True) -> str:
    blob = json.dumps(spec_config(spec, weight_rates), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def model_to_artifact(model: StatVectorModel, meta: dict, weight_rates: bool = True) -> dict:
    """
    :param meta: provenance: trained_on, data_through, row_counts, fit_timestamp
    """
    spec = model.spec
    sub_models = {}
    for target, (features, den) in spec.models.items():
        fit = model.fits[target]
        summary = fit.summary()
        sub_models[target] = {
            "kind": "volume" if den is None else "rate",
            "denominator": den,
            "weighted": bool(weight_rates and den is not None),
            "features": list(fit.features),
            "intercept": _round(fit.coef[0]),
            "coefficients": [_round(c) for c in fit.coef[1:]],
            "std_errors": [_round(s) for s in fit.se],          # [intercept, *features]
            "p_values": [_round(p) for p in summary["p"].to_list()],
            "n": int(fit.n),
            "r2_in_sample": _round(fit.r2),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "position": spec.position,
        "spec_name": f"{spec.position}_SPEC",
        "spec_hash": spec_hash(spec, weight_rates),
        "spec_config": spec_config(spec, weight_rates),
        "window": model.window,
        "weight_rates": weight_rates,
        "rate_priors": {k: _round(v) for k, v in sorted(model.priors.items())},
        "sub_models": sub_models,
        **meta,
    }


def check_compatible(artifact: dict, spec: StatVectorSpec | None = None, weight_rates: bool = True) -> None:
    """Raise ArtifactSpecMismatch unless the artifact was fit for exactly this spec."""
    pos = artifact.get("position")
    spec = spec or STAT_VECTOR_SPECS.get(pos)
    problems = []
    if artifact.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {artifact.get('schema_version')} != {SCHEMA_VERSION}")
    if spec is None:
        raise ArtifactSpecMismatch(f"no in-code spec for position {pos!r}")
    if pos != spec.position:
        problems.append(f"position {pos} != spec {spec.position}")
    if artifact.get("window") != spec.window:
        problems.append(f"window {artifact.get('window')} != spec {spec.window}")
    if artifact.get("weight_rates") != weight_rates:
        problems.append(f"weight_rates {artifact.get('weight_rates')} != {weight_rates}")
    subs = artifact.get("sub_models", {})
    if list(subs) != list(spec.models):
        problems.append(f"sub-models {list(subs)} != spec {list(spec.models)}")
    for target, (features, den) in spec.models.items():
        sub = subs.get(target)
        if sub is None:
            continue
        if sub["features"] != list(features):
            problems.append(f"{target}: features {sub['features']} != spec {list(features)}")
        if sub["denominator"] != den:
            problems.append(f"{target}: denominator {sub['denominator']} != spec {den}")
    if artifact.get("spec_config", {}).get("derivations") != [list(d) for d in spec.derivations]:
        problems.append("derivation chain differs from spec")
    if not problems and artifact.get("spec_hash") != spec_hash(spec, weight_rates):
        problems.append("spec_hash differs from the in-code spec")
    if problems:
        raise ArtifactSpecMismatch(
            f"{pos} artifact does not match the in-code spec ({len(problems)} problem(s)): "
            + "; ".join(problems) + ". Retrain (python -m pipeline.train retrain) instead of applying it."
        )


def artifact_to_model(artifact: dict, spec: StatVectorSpec | None = None) -> StatVectorModel:
    check_compatible(artifact, spec)
    spec = spec or STAT_VECTOR_SPECS[artifact["position"]]
    fits = {}
    for target, sub in artifact["sub_models"].items():
        coef = np.array([sub["intercept"]] + sub["coefficients"], dtype=float)
        fits[target] = OLSFit(target=target, features=list(sub["features"]), coef=coef,
                              se=np.array(sub["std_errors"], dtype=float), n=sub["n"],
                              r2=sub["r2_in_sample"])
    seasons = artifact.get("trained_on", {}).get("seasons", [])
    return StatVectorModel(spec=spec, fits=fits, priors=dict(artifact["rate_priors"]),
                           window=artifact["window"], train_seasons=seasons)


def artifact_path(position: str, directory: Path | str = STAT_VECTOR_DIR) -> Path:
    return Path(directory) / f"{position}.json"


def read_artifact(position: str, directory: Path | str = STAT_VECTOR_DIR) -> dict:
    path = artifact_path(position, directory)
    if not path.exists():
        raise FileNotFoundError(f"no artifact at {path}; run: python -m pipeline.train retrain")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(position: str, directory: Path | str = STAT_VECTOR_DIR) -> StatVectorModel:
    return artifact_to_model(read_artifact(position, directory))


def load_reference_ranges() -> dict:
    if not REFERENCE_RANGES_PATH.exists():
        return {}
    return json.loads(REFERENCE_RANGES_PATH.read_text(encoding="utf-8"))


def coefficient_drift(artifact: dict, reference: dict) -> list[dict]:
    """Compare each coefficient to its 2019-2025 LOSO range. One dict per coefficient."""
    rows = []
    ranges = reference.get("ranges", {}).get(artifact["position"], {})
    for target, sub in artifact["sub_models"].items():
        terms = ["intercept"] + sub["features"]
        values = [sub["intercept"]] + sub["coefficients"]
        for term, value in zip(terms, values):
            rng = ranges.get(target, {}).get(term)
            if rng is None:
                rows.append({"target": target, "term": term, "value": value, "min": None, "max": None,
                             "status": "no reference"})
                continue
            lo, hi = rng["min"], rng["max"]
            width = max(hi - lo, 0.5 * max(abs(lo), abs(hi)), 1e-4)
            if lo <= value <= hi:
                status = "in range"
            elif lo - width <= value <= hi + width:
                status = "outside range"
            else:
                status = "GROSS"
            rows.append({"target": target, "term": term, "value": value, "min": lo, "max": hi, "status": status})
    return rows


def validate(artifacts: dict[str, dict], previous: dict[str, dict], reference: dict) -> tuple[list[str], list[str]]:
    """
    :return: (errors, warnings). Any error blocks the write.
      errors: NaN/inf anywhere numeric, a sub-model below its row floor, a
        coefficient grossly outside its 2019-2025 LOSO range (beyond the range
        by more than its own width), or a previously highly significant
        coefficient (p < 0.01) that moved by > 5 SE and > 50%.
      warnings: coefficient outside its LOSO range (but not gross), moved
        > 25% vs the previous artifact.
    """
    errors, warnings = [], []
    for pos, art in artifacts.items():
        for target, sub in art["sub_models"].items():
            numbers = [sub["intercept"], *sub["coefficients"], *sub["std_errors"], sub["r2_in_sample"]]
            if any(x is None or not math.isfinite(x) for x in numbers):
                errors.append(f"{pos}.{target}: non-finite coefficient / SE / R²")
            floor = MIN_ROWS[sub["kind"]]
            if sub["n"] < floor:
                errors.append(f"{pos}.{target}: {sub['n']} rows < floor {floor}")
        if any(v is not None and not math.isfinite(v) for v in art["rate_priors"].values()):
            errors.append(f"{pos}: non-finite rate prior")

        for d in coefficient_drift(art, reference):
            if d["status"] == "GROSS":
                errors.append(f"{pos}.{d['target']}.{d['term']} = {d['value']:.4g} grossly outside "
                              f"2019-2025 range [{d['min']:.4g}, {d['max']:.4g}]")
            elif d["status"] == "outside range":
                warnings.append(f"{pos}.{d['target']}.{d['term']} = {d['value']:.4g} outside "
                                f"2019-2025 range [{d['min']:.4g}, {d['max']:.4g}]")

        prev = previous.get(pos)
        if prev and prev.get("spec_hash") == art["spec_hash"]:
            for target, sub in art["sub_models"].items():
                psub = prev["sub_models"][target]
                terms = ["intercept"] + sub["features"]
                new_vals = [sub["intercept"]] + sub["coefficients"]
                old_vals = [psub["intercept"]] + psub["coefficients"]
                for i, (term, new, old) in enumerate(zip(terms, new_vals, old_vals)):
                    delta = abs(new - old)
                    se, p = psub["std_errors"][i], psub["p_values"][i]
                    if p is not None and p < 0.01 and delta > 5 * se and delta > 0.5 * abs(old):
                        errors.append(f"{pos}.{target}.{term}: {old:.4g} -> {new:.4g} "
                                      f"(> 5 SE and > 50% vs previous artifact)")
                    elif abs(old) > 1e-9 and delta > 0.25 * abs(old):
                        warnings.append(f"{pos}.{target}.{term}: {old:.4g} -> {new:.4g} (> 25% vs previous)")
    return errors, warnings


def read_all(directory: Path | str = STAT_VECTOR_DIR) -> dict[str, dict]:
    out = {}
    for pos in STAT_VECTOR_SPECS:
        path = artifact_path(pos, directory)
        if path.exists():
            out[pos] = json.loads(path.read_text(encoding="utf-8"))
    return out


def write_all(artifacts: dict[str, dict], metrics_md: str, directory: Path | str = STAT_VECTOR_DIR,
              reference: dict | None = None) -> tuple[list[str], dict[str, Path]]:
    """
    Safe overwrite: write everything to a temp dir, validate, then swap each
    file into place, keeping one previous generation as <POS>.json.prev.
    Raises ArtifactValidationError (leaving existing files untouched) on any
    validation error. Returns (warnings, paths).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory.parent / "MODEL_METRICS.md"
    tmp = directory.parent / f".tmp-{directory.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for pos, art in artifacts.items():
            (tmp / f"{pos}.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
        (tmp / "MODEL_METRICS.md").write_text(metrics_md, encoding="utf-8")

        # Validate what was actually written (round-trip through JSON).
        written = {pos: json.loads((tmp / f"{pos}.json").read_text(encoding="utf-8")) for pos in artifacts}
        errors, warnings = validate(written, read_all(directory),
                                    load_reference_ranges() if reference is None else reference)
        if errors:
            raise ArtifactValidationError(
                f"{len(errors)} validation error(s); existing artifacts left untouched:\n  " + "\n  ".join(errors))

        paths = {}
        for pos in artifacts:
            final = artifact_path(pos, directory)
            if final.exists():
                os.replace(final, final.with_suffix(".json.prev"))
            os.replace(tmp / f"{pos}.json", final)
            paths[pos] = final
        os.replace(tmp / "MODEL_METRICS.md", metrics_path)
        paths["metrics"] = metrics_path
        return warnings, paths
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
