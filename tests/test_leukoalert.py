"""Synthetic fixtures test conversion and provenance, not clinical performance."""
import csv
import io
import json
import zipfile
import pytest
from hemasight.research.leukoalert import SOURCE_FILES, prepare_archive, sha256_file
from hemasight.research.schema import COLUMN_MAP


def make_archive(tmp_path, *, label="healthy", drop_header=None):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in SOURCE_FILES:
            fields = ["sample_id", *(value[0] for value in COLUMN_MAP.values()), "leukemia_label"]
            if drop_header:
                fields.remove(drop_header)
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            row = {field: "1" for field in fields}
            row.update(sample_id="synthetic-sample", leukemia_label=label)
            for column, value in [("HGB(g/L)", "140"), ("MCHC(g/L)", "320"),
                                  ("LYMPH%(%)", "40"), ("LYMPH#(10^9/L)", "2.4"),
                                  ("WBC(10^9/L)", ""), ("RDW-CV(%)", "invalid")]:
                if column in row:
                    row[column] = value
            writer.writerow(row)
            archive.writestr(name, stream.getvalue())
        archive.writestr("../../unrelated.py", "raise RuntimeError('Must never run or extract')")
    return path


def test_preserves_cohorts_and_units_without_inventing_history(tmp_path):
    archive = make_archive(tmp_path)
    output = tmp_path / "prepared"
    manifest = prepare_archive(archive, output, expected_sha256=sha256_file(archive))
    with (output / "features.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11
    assert {(r["cohort"], r["site"]) for r in rows} == set(SOURCE_FILES.values())
    for feature, value in [("hemoglobin", 14), ("mchc", 32), ("lymphocytes", 40), ("lymphocytes_absolute", 2.4)]:
        assert {float(r[feature]) for r in rows} == {value}
    assert all(r["wbc"] == "" for r in rows)
    assert not {"patient_id", "date", "wbc_trend"} & rows[0].keys()
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert all(c["numeric_issues"]["rdw_cv"] == {"invalid_numeric": 1} for c in manifest["cohorts"])
    assert not manifest["verified_official_archive"]
    assert not (tmp_path / "unrelated.py").exists()
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["prepared_csv_sha256"] == sha256_file(output / "features.csv")


def test_rejects_wrong_checksum_without_creating_output(tmp_path):
    archive = make_archive(tmp_path)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_archive(archive, output)
    assert not output.exists()


@pytest.mark.parametrize("options, message", [({"label": "at-risk"}, "Unknown leukemia label"),
                                             ({"drop_header": "HGB(g/L)"}, "Missing columns")])
def test_bad_source_cannot_leave_a_partial_dataset(tmp_path, options, message):
    archive = make_archive(tmp_path, **options)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=message):
        prepare_archive(archive, output, expected_sha256=sha256_file(archive))
    assert not output.exists()


def test_preparation_never_overwrites_an_existing_dataset(tmp_path):
    archive = make_archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(FileExistsError):
        prepare_archive(archive, output, expected_sha256=sha256_file(archive))
    assert sentinel.read_text() == "existing work"
