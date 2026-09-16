"""Prepare the pinned public LeukoAlert archive without executing its source code.

No dates or patient histories are inferred. All preprocessing learned from data
(including imputation) belongs in the benchmark's training pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

from hemasight.research.schema import COLUMN_MAP, FEATURE_SETS, SCHEMA_VERSION

SOURCE_URL = "https://data.mendeley.com/datasets/vc7kwnyppz/1"
DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/vc7kwnyppz/files/2edb7026-6a23-493a-8d75-78f79fdb12b3/file_downloaded"
ARCHIVE_SHA256 = "3c09b869867cc1b7298943ad332d7d60515b9c3dba14c06e5e3338e67318ed05"
PREFIX = "LeukoAlert-project/leuk vs HC/dataset/"
SOURCE_FILES = {
    **{f"{PREFIX}training/train_site_{site}.csv": ("development", site) for site in "ABC"},
    **{f"{PREFIX}validation/valid_site_{site}.csv": ("validation", site) for site in "ABC"},
    **{f"{PREFIX}validation/valid_site_{site}.csv": ("external", site) for site in "DEFG"},
    f"{PREFIX}test/test_site_true_world.csv": ("test", "real_world"),
}
LIMITATIONS = [
    "Screening labels describe the released record; they do not define future cancer incidence.",
    "No explicit patient identifier or test/diagnosis dates are released in these CSVs.",
    "Sample IDs are not patient IDs; patient overlap and repeated-measure independence are unverified.",
    "The archive contains 446663 binary-task records versus 446558 reported by the article (difference 105).",
    "The binary CSV schema has 76 columns; this benchmark uses explicit 5/15-feature sets, not a reproduction of the paper's 72-feature model.",
    "Site labels come from filenames. The real_world filename is not a newly identified hospital.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        if sha256_file(output) != ARCHIVE_SHA256:
            raise ValueError("Existing archive checksum does not match LeukoAlert v1; it was not overwritten.")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".download", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with urlopen(DOWNLOAD_URL, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256_file(temporary) != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive checksum mismatch; do not use it as the pinned release.")
        # Exclusive creation avoids accidentally replacing an existing user file.
        with temporary.open("rb") as source, output.open("xb") as dest:
            shutil.copyfileobj(source, dest)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _numeric(raw: str | None, multiplier: float) -> tuple[float | str, str | None]:
    if raw is None or raw.strip().lower() in {"", "nan", "na", "n/a", "null", "none"}:
        return "", "missing"
    try:
        value = float(raw) * multiplier
    except ValueError:
        return "", "invalid_numeric"
    if not math.isfinite(value):
        return "", "nonfinite"
    return value, None


def prepare_archive(
    archive_path: str | Path,
    output: str | Path,
    *,
    expected_sha256: str = ARCHIVE_SHA256,
) -> dict:
    """Write a new dataset directory; alternate hashes are only for explicit fixtures."""
    archive_path, output = Path(archive_path), Path(output)
    source_hash = sha256_file(archive_path)
    if source_hash != expected_sha256:
        raise ValueError("Archive SHA-256 does not match the expected release.")
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing datasets are never overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".prepare-") as tmp:
        staging = Path(tmp)
        destination = staging / "features.csv"
        audit = []
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            absent = sorted(set(SOURCE_FILES) - set(names))
            if absent:
                raise ValueError(f"Archive is missing expected cohort files: {absent}")
            if any(names.count(name) != 1 for name in SOURCE_FILES):
                raise ValueError("Archive contains duplicate cohort file entries.")
            with destination.open("w", encoding="utf-8", newline="") as dest:
                writer = csv.DictWriter(dest, fieldnames=["sample_id", "site", "cohort", "target", *COLUMN_MAP])
                writer.writeheader()
                for name, (cohort, site) in SOURCE_FILES.items():
                    record_count = 0
                    labels = Counter()
                    issues = {feature: Counter() for feature in COLUMN_MAP}
                    seen_sample_ids = set()
                    duplicate_sample_ids = 0
                    with archive.open(name) as raw:
                        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                        required = {"sample_id", "leukemia_label", *(spec[0] for spec in COLUMN_MAP.values())}
                        missing = required - set(reader.fieldnames or [])
                        if missing:
                            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")
                        if len(reader.fieldnames) != len(set(reader.fieldnames)):
                            raise ValueError(f"Duplicate columns in {name}")
                        for number, row in enumerate(reader, start=2):
                            if None in row or any(v is None for v in row.values()):
                                raise ValueError(f"Malformed CSV record in {name}, line {number}")
                            label = row["leukemia_label"].strip()
                            if label not in {"healthy", "leukemia"}:
                                raise ValueError(f"Unknown leukemia label in {name}, line {number}")
                            sample_id = row["sample_id"].strip()
                            if not sample_id:
                                raise ValueError(f"Missing sample ID in {name}, line {number}")
                            duplicate_sample_ids += sample_id in seen_sample_ids
                            seen_sample_ids.add(sample_id)
                            # File namespace preserves provenance, not patient identity.
                            prepared = {"sample_id": f"{Path(name).stem}:{sample_id}", "site": site,
                                        "cohort": cohort, "target": int(label == "leukemia")}
                            for feature, (source, multiplier, unit) in COLUMN_MAP.items():
                                prepared[feature], issue = _numeric(row[source], multiplier)
                                if issue:
                                    issues[feature][issue] += 1
                            writer.writerow(prepared)
                            record_count += 1
                            labels[label] += 1
                    if not record_count:
                        raise ValueError(f"Empty cohort file: {name}")
                    audit.append({"file": name, "cohort": cohort, "site": site, "rows": record_count,
                                  "labels": dict(labels), "duplicate_sample_ids_within_file": duplicate_sample_ids,
                                  "input_column_count": len(reader.fieldnames),
                                  "numeric_issues": {k: dict(v) for k, v in issues.items()}})
        total = sum(item["rows"] for item in audit)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": SOURCE_URL, "source_doi": "10.17632/vc7kwnyppz.1", "source_license": "CC BY 4.0",
            "source_archive_sha256": source_hash, "prepared_csv_sha256": sha256_file(destination),
            "verified_official_archive": source_hash == ARCHIVE_SHA256,
            "records": total, "published_record_total": 446558,
            "difference_from_published_total": total - 446558,
            "feature_sets": FEATURE_SETS,
            "column_mapping": {key: {"source": value[0], "multiplier": value[1], "unit": value[2]} for key, value in COLUMN_MAP.items()},
            "cohorts": audit, "limitations": LIMITATIONS,
            "processing": "Unit conversion and audited missing/invalid-number parsing only; no imputation, scaling, resampling, or clinical range filtering.",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lines = ["# LeukoAlert preparation audit", "", f"Source: [{SOURCE_URL}]({SOURCE_URL})", "",
                 f"Records read: **{total:,}**. Source checksum: `{source_hash}`.", "",
                 "| Cohort | Site | Records | Leukemia | Negative label |", "| --- | --- | ---: | ---: | ---: |"]
        lines += [f"| {a['cohort']} | {a['site']} | {a['rows']:,} | {a['labels'].get('leukemia', 0):,} | {a['labels'].get('healthy', 0):,} |" for a in audit]
        lines += ["", "## Interpretation limits", "", *(f"- {note}" for note in LIMITATIONS), "",
                  "Numeric parsing issues and exact unit mappings are in manifest.json. Missing values are left for training-only imputation.", ""]
        (staging / "audit.md").write_text("\n".join(lines), encoding="utf-8")
        output.mkdir(exist_ok=False)
        for path in staging.iterdir():
            shutil.move(str(path), output / path.name)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify the pinned public archive")
    download.add_argument("--output", default="data/raw/LeukoAlert-project.zip")
    prepare = commands.add_parser("prepare", help="Prepare all source partitions and write an audit")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--output", required=True, help="New directory for features.csv, manifest.json, audit.md")
    args = parser.parse_args()
    if args.command == "download":
        print(download_archive(args.output))
    else:
        manifest = prepare_archive(args.archive, args.output)
        print(f"Prepared {manifest['records']:,} records in {args.output}. No patient linkage or prediction dates inferred.")


if __name__ == "__main__":
    main()
