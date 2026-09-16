"""Describe a previously evaluated dataset without fitting models or imputing values."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .benchmark import _load_dataset
from .profile_schema import DatasetProfile
from .schema import COLUMN_MAP


def summarize_group(frame, cohort, site=None):
    features = {}
    for name in COLUMN_MAP:
        values = frame[name].dropna()
        quantiles = values.quantile([.05, .25, .5, .75, .95]).tolist() if len(values) else [None] * 5
        features[name] = {
            "observed": len(values), "missing": len(frame) - len(values),
            **dict(zip(("p05", "q1", "median", "q3", "p95"), quantiles)),
        }
    return {"cohort": cohort, "scope": "overall" if site is None else "site", "site": site,
            "n": len(frame), "n_positive": int(frame.target.sum()), "features": features}


def build_profile(dataset: Path, run_directory: Path) -> Path:
    """Bind the profile to a completed run; only inspect its already-evaluated cohorts."""
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("A completed benchmark is required")
    destination = run_directory / "profile.json"
    if destination.exists():
        raise FileExistsError("The profile already exists; preserve the saved report")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != manifest["dataset"]["sha256"]:
        raise ValueError("Dataset checksum differs from the completed benchmark")
    cohorts = {"development", *manifest["evaluated_cohorts"]}
    if not manifest["evaluate_holdouts"] and cohorts - {"development", "validation"}:
        raise ValueError("Holdout inspection was not enabled in the completed run")
    frame = _load_dataset(dataset, ["expanded_cbc"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != digest:
        raise ValueError("Dataset changed during profile generation")
    if len(frame) != manifest["dataset"]["rows"]:
        raise ValueError("Dataset size differs from benchmark")
    groups = []
    for cohort in ("development", "validation", "external", "test"):
        if cohort not in cohorts:
            continue
        subset = frame.loc[frame.cohort == cohort]
        expected = manifest["cohort_counts"][cohort]
        if len(subset) != expected["n"] or int(subset.target.sum()) != expected["n_positive"]:
            raise ValueError("Cohort counts differ from benchmark")
        groups.append(summarize_group(subset, cohort))
        for site, site_frame in subset.groupby("site", sort=True):
            groups.append(summarize_group(site_frame, cohort, str(site)))
    report = DatasetProfile(
        schema_version="cbc_profile_v1", dataset_sha256=digest,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        units={name: spec[2] for name, spec in COLUMN_MAP.items()}, groups=groups,
    )
    with destination.open("x", encoding="utf-8") as output:
        output.write(report.model_dump_json(indent=2))
        output.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    print(build_profile(args.dataset, args.run_directory))


if __name__ == "__main__":
    main()
