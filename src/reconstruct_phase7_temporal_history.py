#!/usr/bin/env python3
"""
Phase 7.5c — Corrected Temporal Reconstruction and Validation

Purpose
-------
Reconstruct the portable environmental-measurement chronology conservatively
from the original HydroGrowNet Excel workbooks after P7.5b established that
Excel auto-converted some DD/MM/YYYY entries into misleading datetime objects.

Scientific rule used here
-------------------------
The source sequence contains explicit unambiguous DD/MM/YYYY strings (for
example days > 12) surrounding Excel datetime cells. Therefore Excel datetime
cells are reconstructed from their DISPLAYED month/day components as:

    corrected day   = Excel datetime.month
    corrected month = Excel datetime.day
    corrected year  = Excel datetime.year

Example:
    Excel object 2024-10-03  -> source date 10/03/2024 -> 2024-03-10

String dates are parsed as DD/MM/YYYY after normalizing /, \\, -, and . separators.

This script:
- does NOT modify the raw ZIP/workbooks;
- does NOT infer harvest dates;
- does NOT construct harvest-relative exposure windows;
- does NOT calculate shift severity;
- does NOT freeze target/features/domains/augmentation/models;
- does NOT train any model.

Outputs are diagnostic/reconstructed temporal tables only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "phase7" / "raw"
OUT = ROOT / "results" / "experiment_7" / "temporal_reconstruction"

OUT.mkdir(parents=True, exist_ok=True)

PORTABLE_HINT = "water quality parametersportabl"


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_zip() -> Path:
    preferred = RAW_DIR / "all_months_sensory_data.zip"
    if preferred.exists():
        return preferred

    zips = sorted(RAW_DIR.glob("*.zip"))
    if len(zips) == 1:
        return zips[0]
    if not zips:
        raise FileNotFoundError(f"No ZIP archive found in {RAW_DIR}")
    raise RuntimeError(
        "Multiple ZIP archives found and preferred archive is absent:\n"
        + "\n".join(f"  - {p.name}" for p in zips)
    )


def experiment_from_name(name: str) -> str | None:
    m = re.search(r"EXP[.\s_-]*(\d+)", Path(name).name, flags=re.I)
    return f"EXP{int(m.group(1))}" if m else None


def find_portable_sheet(names: list[str]) -> str | None:
    for name in names:
        if name.lower().strip() == PORTABLE_HINT:
            return name
    for name in names:
        if "portabl" in name.lower():
            return name
    return None


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return clean(value) == ""


def normalize_date_string(text: str) -> str:
    s = text.strip()
    s = s.replace("\\", "/").replace("-", "/").replace(".", "/")
    s = re.sub(r"/+", "/", s)
    return s


def parse_ddmmyyyy_string(value: Any) -> pd.Timestamp:
    s = normalize_date_string(clean(value))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if not m:
        return pd.NaT

    d, mth, y = map(int, m.groups())
    if y < 100:
        y += 2000

    try:
        return pd.Timestamp(year=y, month=mth, day=d)
    except Exception:
        return pd.NaT


def reconstruct_date(value: Any) -> tuple[pd.Timestamp, str, str]:
    """
    Returns:
        corrected_timestamp,
        reconstruction_method,
        note
    """
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        # Evidence from P7.5b:
        # Excel interpreted source DD/MM/YYYY as m/d/yyyy when day <= 12.
        # Thus Excel's month is the original day and Excel's day is the
        # original month.
        try:
            corrected = pd.Timestamp(
                year=value.year,
                month=value.day,
                day=value.month,
            )
            return (
                corrected,
                "excel_datetime_component_swap",
                (
                    f"Excel={value:%Y-%m-%d}; reconstructed as "
                    f"day={value.month}, month={value.day}, year={value.year}"
                ),
            )
        except Exception as exc:
            return pd.NaT, "excel_datetime_component_swap_failed", str(exc)

    if isinstance(value, date):
        try:
            corrected = pd.Timestamp(
                year=value.year,
                month=value.day,
                day=value.month,
            )
            return (
                corrected,
                "excel_date_component_swap",
                (
                    f"Excel={value.isoformat()}; reconstructed as "
                    f"day={value.month}, month={value.day}, year={value.year}"
                ),
            )
        except Exception as exc:
            return pd.NaT, "excel_date_component_swap_failed", str(exc)

    parsed = parse_ddmmyyyy_string(value)
    if pd.notna(parsed):
        return (
            parsed,
            "explicit_ddmmyyyy_string",
            f"raw={clean(value)!r}",
        )

    return pd.NaT, "unparsed", f"raw={clean(value)!r}"


def looks_like_raw_date(value: Any) -> bool:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return True
    return pd.notna(parse_ddmmyyyy_string(value))


def row_has_measurements(df: pd.DataFrame, row: int) -> bool:
    if df.shape[1] <= 1:
        return False
    for c in range(1, df.shape[1]):
        v = df.iat[row, c]
        if is_blank(v):
            continue
        s = clean(v)
        if s in {"-", "–", "—"}:
            continue
        try:
            float(v)
            return True
        except Exception:
            continue
    return False


def fmt_date(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def main() -> int:
    archive = find_zip()
    archive_hash = sha256_file(archive)

    records: list[dict[str, Any]] = []
    experiment_summary: list[dict[str, Any]] = []
    errors: list[str] = []

    with zipfile.ZipFile(archive, "r") as zf:
        excel_members = [
            name
            for name in zf.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".xlsx", ".xlsm"}
            and not Path(name).name.startswith("~$")
        ]

        for member in sorted(excel_members):
            experiment = experiment_from_name(member)
            if experiment not in {"EXP1", "EXP2", "EXP3"}:
                continue

            try:
                raw_bytes = zf.read(member)
                xls = pd.ExcelFile(BytesIO(raw_bytes), engine="openpyxl")
                portable = find_portable_sheet(xls.sheet_names)

                if portable is None:
                    errors.append(f"{experiment}: portable sheet not found")
                    continue

                df = pd.read_excel(
                    BytesIO(raw_bytes),
                    sheet_name=portable,
                    header=None,
                    dtype=object,
                    engine="openpyxl",
                )

                exp_records = []

                # The portable date field established in P7.4/P7.5/P7.5b is C0.
                for r in range(df.shape[0]):
                    raw = df.iat[r, 0]
                    if not looks_like_raw_date(raw):
                        continue

                    corrected, method, note = reconstruct_date(raw)

                    rec = {
                        "experiment": experiment,
                        "workbook": Path(member).name,
                        "sheet": portable,
                        "row_zero_based": r,
                        "excel_row": r + 1,
                        "raw_value": clean(raw),
                        "raw_repr": repr(raw),
                        "raw_python_type": type(raw).__name__,
                        "corrected_date": fmt_date(corrected),
                        "reconstruction_method": method,
                        "reconstruction_note": note,
                        "row_has_numeric_measurements": row_has_measurements(df, r),
                    }
                    records.append(rec)
                    exp_records.append(rec)

                exp_df = pd.DataFrame(exp_records)

                if exp_df.empty:
                    experiment_summary.append(
                        {
                            "experiment": experiment,
                            "n_date_rows": 0,
                            "n_parsed": 0,
                            "n_unparsed": 0,
                            "first_date": "",
                            "last_date": "",
                            "backward_steps": np.nan,
                            "duplicate_dates": np.nan,
                            "one_day_steps": np.nan,
                            "positive_gaps_gt_1_day": np.nan,
                            "largest_positive_gap_days": np.nan,
                            "status": "FAIL_NO_DATES",
                        }
                    )
                    continue

                parsed = pd.to_datetime(
                    exp_df["corrected_date"],
                    errors="coerce",
                )
                diffs = parsed.diff().dt.days

                n_unparsed = int(parsed.isna().sum())
                backward = int((diffs < 0).sum())
                duplicates = int((diffs == 0).sum())
                one_day = int((diffs == 1).sum())
                gaps = diffs[diffs > 1]
                n_gaps = int(gaps.notna().sum())
                largest_gap = (
                    int(gaps.max()) if len(gaps) and pd.notna(gaps.max()) else 0
                )

                status = (
                    "PASS_MONOTONIC"
                    if n_unparsed == 0 and backward == 0
                    else "REVIEW_REQUIRED"
                )

                experiment_summary.append(
                    {
                        "experiment": experiment,
                        "n_date_rows": int(len(exp_df)),
                        "n_parsed": int(parsed.notna().sum()),
                        "n_unparsed": n_unparsed,
                        "first_date": (
                            fmt_date(parsed.iloc[0])
                            if len(parsed) and pd.notna(parsed.iloc[0])
                            else ""
                        ),
                        "last_date": (
                            fmt_date(parsed.iloc[-1])
                            if len(parsed) and pd.notna(parsed.iloc[-1])
                            else ""
                        ),
                        "backward_steps": backward,
                        "duplicate_dates": duplicates,
                        "one_day_steps": one_day,
                        "positive_gaps_gt_1_day": n_gaps,
                        "largest_positive_gap_days": largest_gap,
                        "status": status,
                    }
                )

            except Exception as exc:
                errors.append(
                    f"{experiment} {Path(member).name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    rec_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(experiment_summary)

    # Add within-experiment temporal diagnostics to each row.
    if not rec_df.empty:
        rec_df["corrected_date_ts"] = pd.to_datetime(
            rec_df["corrected_date"],
            errors="coerce",
        )
        rec_df["step_days_from_previous"] = (
            rec_df.groupby("experiment", sort=False)["corrected_date_ts"]
            .diff()
            .dt.days
        )
        rec_df["is_backward_step"] = rec_df["step_days_from_previous"] < 0
        rec_df["is_duplicate_date"] = rec_df["step_days_from_previous"] == 0
        rec_df["is_gap_gt_1_day"] = rec_df["step_days_from_previous"] > 1
        rec_df.drop(columns=["corrected_date_ts"], inplace=True)

    expected_experiments = {"EXP1", "EXP2", "EXP3"}
    found_experiments = (
        set(summary_df["experiment"].tolist())
        if not summary_df.empty
        else set()
    )

    all_three = found_experiments == expected_experiments
    all_parsed = (
        not summary_df.empty
        and int(summary_df["n_unparsed"].sum()) == 0
    )
    no_backward = (
        not summary_df.empty
        and int(summary_df["backward_steps"].sum()) == 0
    )
    no_duplicates = (
        not summary_df.empty
        and int(summary_df["duplicate_dates"].sum()) == 0
    )

    chronology_pass = (
        len(errors) == 0
        and all_three
        and all_parsed
        and no_backward
        and no_duplicates
    )

    report: list[str] = []
    report.append("PHASE 7.5c — CORRECTED TEMPORAL RECONSTRUCTION AND VALIDATION")
    report.append("=" * 108)
    report.append(f"Archive: {archive}")
    report.append(f"Archive SHA256: {archive_hash}")
    report.append("")
    report.append("SCIENTIFIC BASIS")
    report.append("-" * 108)
    report.append(
        "P7.5b showed continuous explicit DD/MM/YYYY sequences around Excel datetime cells."
    )
    report.append(
        "For Excel datetime cells, the original source day is reconstructed from Excel.month "
        "and the original source month from Excel.day."
    )
    report.append(
        "Explicit string dates are parsed as DD/MM/YYYY after separator normalization."
    )
    report.append(
        "This corrects Excel auto-conversion only; it does not infer harvest dates."
    )
    report.append("")

    report.append("EXPERIMENT-LEVEL CHRONOLOGY")
    report.append("-" * 108)
    if summary_df.empty:
        report.append("NO EXPERIMENT SUMMARIES PRODUCED")
    else:
        for row in summary_df.itertuples(index=False):
            report.append(
                f"{row.experiment}: rows={row.n_date_rows}, parsed={row.n_parsed}, "
                f"unparsed={row.n_unparsed}, first={row.first_date}, last={row.last_date}, "
                f"backward={row.backward_steps}, duplicates={row.duplicate_dates}, "
                f"1-day steps={row.one_day_steps}, gaps>1day={row.positive_gaps_gt_1_day}, "
                f"largest_gap={row.largest_positive_gap_days}, status={row.status}"
            )
    report.append("")

    report.append("RECONSTRUCTED SOURCE SEQUENCES")
    report.append("-" * 108)
    if rec_df.empty:
        report.append("NONE")
    else:
        for exp, g in rec_df.groupby("experiment", sort=True):
            report.append(f"{exp}:")
            for row in g.itertuples(index=False):
                step = row.step_days_from_previous
                step_txt = "START" if pd.isna(step) else f"{int(step):+d} day(s)"
                report.append(
                    f"  source R{row.row_zero_based}: raw={row.raw_repr} -> "
                    f"{row.corrected_date} [{row.reconstruction_method}] | step={step_txt}"
                )
    report.append("")

    report.append("GAPS / IRREGULARITIES")
    report.append("-" * 108)
    irregular = (
        rec_df[
            rec_df["is_backward_step"]
            | rec_df["is_duplicate_date"]
            | rec_df["is_gap_gt_1_day"]
        ]
        if not rec_df.empty
        else pd.DataFrame()
    )
    if irregular.empty:
        report.append("No backward steps, duplicate dates, or >1-day positive gaps detected.")
    else:
        for row in irregular.itertuples(index=False):
            report.append(
                f"{row.experiment} source R{row.row_zero_based}: "
                f"{row.corrected_date}, step={row.step_days_from_previous} days, "
                f"backward={row.is_backward_step}, duplicate={row.is_duplicate_date}, "
                f"gap_gt_1_day={row.is_gap_gt_1_day}"
            )
    report.append("")

    report.append("VALIDATION")
    report.append("-" * 108)
    report.append(f"Workbook processing errors: {len(errors)}")
    for err in errors:
        report.append(f"  - {err}")
    report.append(f"Expected experiments recovered: {all_three}")
    report.append(f"All date rows parsed: {all_parsed}")
    report.append(f"No backward chronological steps: {no_backward}")
    report.append(f"No duplicate reconstructed dates: {no_duplicates}")
    report.append(
        "Chronology validation: " + ("PASS" if chronology_pass else "REVIEW REQUIRED")
    )
    report.append("")

    report.append("SCIENTIFIC LIMITS")
    report.append("-" * 108)
    report.append(
        "1. PASS validates source-order environmental measurement chronology only."
    )
    report.append(
        "2. No explicit harvest date is inferred or frozen by this script."
    )
    report.append(
        "3. No claim is made that every calendar day has a measurement."
    )
    report.append(
        "4. No harvest-relative exposure window is constructed."
    )
    report.append(
        "5. No shift family or shift severity is calculated."
    )
    report.append("")

    report.append("RESEARCH LOCK STATUS")
    report.append("-" * 108)
    report.append(
        "Environmental source chronology validated: "
        + ("YES" if chronology_pass else "NO")
    )
    report.append("Harvest dates frozen: NO")
    report.append("Primary target frozen: NO")
    report.append("Feature set frozen: NO")
    report.append("Domain definition frozen: NO")
    report.append("Shift family frozen: NO")
    report.append("Shift severity frozen: NO")
    report.append("Augmentation frozen: NO")
    report.append("Model training performed: NO")

    rec_csv = OUT / "phase7_corrected_portable_dates.csv"
    summary_csv = OUT / "phase7_temporal_validation_summary.csv"
    report_path = OUT / "phase7_temporal_reconstruction_report.txt"
    manifest_path = OUT / "phase7_temporal_reconstruction_manifest.json"

    rec_df.to_csv(rec_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "phase": "7.5c",
        "purpose": "corrected environmental measurement chronology",
        "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "date_rule": {
            "strings": "DD/MM/YYYY after separator normalization",
            "excel_datetime": (
                "corrected_day=excel.month; corrected_month=excel.day; "
                "corrected_year=excel.year"
            ),
        },
        "experiments_recovered": sorted(found_experiments),
        "chronology_validation_pass": bool(chronology_pass),
        "processing_errors": errors,
        "harvest_dates_inferred": False,
        "shift_severity_calculated": False,
        "model_training_performed": False,
        "phase7_design_frozen": False,
        "outputs": [
            str(rec_csv.relative_to(ROOT)),
            str(summary_csv.relative_to(ROOT)),
            str(report_path.relative_to(ROOT)),
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 90)
    print("PHASE 7.5c — CORRECTED TEMPORAL RECONSTRUCTION")
    print("=" * 90)
    print(f"Experiments recovered: {sorted(found_experiments)}")
    print(f"Date rows reconstructed: {len(rec_df)}")
    print(f"Processing errors: {len(errors)}")
    print(
        "Chronology validation: "
        + ("PASS" if chronology_pass else "REVIEW REQUIRED")
    )
    print(f"Report: {report_path}")
    print("")
    print("NO HARVEST DATE WAS INFERRED.")
    print("NO MODEL TRAINING WAS PERFORMED.")

    return 0 if chronology_pass else 1


if __name__ == "__main__":
    sys.exit(main())
