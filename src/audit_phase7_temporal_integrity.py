#!/usr/bin/env python3
"""
Phase 7.5 — Temporal Integrity Audit

Purpose
-------
Audit the real HydroGrowNet Phase-7 environmental histories BEFORE any
modeling-design freeze or ML training.

Checks:
1. Raw date strings/cells from each portable worksheet.
2. Date interpretation using day-first parsing.
3. Chronological order and suspicious date jumps.
4. Morning/afternoon measurement-time structure.
5. Domain-level missingness for pH, EC, TDS, water temperature.
6. Harvest-date discovery from harvest worksheets.
7. Whether reconstructed environmental measurements occur on/before harvest.
8. Whether all 18 treatment-replicate domains remain represented.

This script is diagnostic only:
- no model training
- no augmentation
- no imputation
- no target/predictor/domain/severity freeze
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "phase7" / "raw"
LINKAGE_DIR = ROOT / "results" / "experiment_7" / "environment_linkage"
OUT = ROOT / "results" / "experiment_7" / "temporal_integrity"

RECORDS_FILE = LINKAGE_DIR / "phase7_domain_environment_records.csv"

PORTABLE_HINT = "water quality parametersportabl"
HARVEST_HINTS = ("harvest", "measurement")

ENV_VARS = ["ph", "ec", "tds", "water_temperature"]


def clean(v: Any) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


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
    raise RuntimeError("Multiple ZIP files found; expected all_months_sensory_data.zip.")


def experiment_from_name(name: str) -> str | None:
    m = re.search(r"EXP[.\s_-]*(\d+)", Path(name).name, flags=re.I)
    return f"EXP{int(m.group(1))}" if m else None


def parse_date_dayfirst(v: Any) -> pd.Timestamp:
    if pd.isna(v):
        return pd.NaT
    if isinstance(v, pd.Timestamp):
        return v.normalize()
    if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):
        try:
            return pd.Timestamp(v).normalize()
        except Exception:
            pass
    s = clean(v).replace("\\", "/")
    if not s:
        return pd.NaT
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def parse_date_monthfirst(v: Any) -> pd.Timestamp:
    if pd.isna(v):
        return pd.NaT
    s = clean(v).replace("\\", "/")
    if not s:
        return pd.NaT
    return pd.to_datetime(s, dayfirst=False, errors="coerce")


def date_ambiguity(v: Any) -> bool:
    """True for numeric dates where both first two fields could be month/day."""
    s = clean(v).replace("\\", "/").replace("-", "/")
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*", s)
    if not m:
        return False
    a, b = int(m.group(1)), int(m.group(2))
    return 1 <= a <= 12 and 1 <= b <= 12 and a != b


def find_sheet(names: list[str], portable: bool) -> str | None:
    if portable:
        for n in names:
            if n.lower().strip() == PORTABLE_HINT:
                return n
        for n in names:
            if "portabl" in n.lower():
                return n
        return None

    candidates = [
        n for n in names
        if "harvest" in n.lower()
        or ("plant" in n.lower() and "measurement" in n.lower())
    ]
    return candidates[0] if candidates else None


def raw_portable_date_audit(df: pd.DataFrame, experiment: str) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows

    for r in range(df.shape[0]):
        raw = df.iat[r, 0] if df.shape[1] else np.nan
        txt = clean(raw)
        if not txt:
            continue

        d_day = parse_date_dayfirst(raw)
        d_month = parse_date_monthfirst(raw)

        # Retain anything parseable as a date; this avoids guessing header row count.
        if pd.isna(d_day) and pd.isna(d_month):
            continue

        rows.append({
            "experiment": experiment,
            "source_row_zero_based": r,
            "raw_date_value": txt,
            "dayfirst_date": d_day.strftime("%Y-%m-%d") if pd.notna(d_day) else "",
            "monthfirst_date": d_month.strftime("%Y-%m-%d") if pd.notna(d_month) else "",
            "ambiguous_numeric_date": date_ambiguity(raw),
            "interpretations_differ": (
                pd.notna(d_day) and pd.notna(d_month) and d_day.normalize() != d_month.normalize()
            ),
        })
    return rows


def infer_sequence_quality(date_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for exp, g in date_df.groupby("experiment", sort=True):
        g = g.sort_values("source_row_zero_based")
        dates = pd.to_datetime(g["dayfirst_date"], errors="coerce").dropna()
        diffs = dates.diff().dt.days.dropna()

        results.append({
            "experiment": exp,
            "n_raw_date_rows": int(len(g)),
            "n_dayfirst_parsed": int(pd.to_datetime(g["dayfirst_date"], errors="coerce").notna().sum()),
            "n_ambiguous_numeric": int(g["ambiguous_numeric_date"].sum()),
            "n_differing_interpretations": int(g["interpretations_differ"].sum()),
            "dayfirst_first": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
            "dayfirst_last": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
            "negative_steps_in_source_order": int((diffs < 0).sum()),
            "zero_day_steps": int((diffs == 0).sum()),
            "max_forward_gap_days": int(diffs.max()) if len(diffs) else 0,
            "median_forward_gap_days": float(diffs[diffs >= 0].median()) if (diffs >= 0).any() else np.nan,
        })
    return pd.DataFrame(results)


def linkage_record_audit(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = records.copy()
    records["date_parsed"] = pd.to_datetime(records["date"], errors="coerce")

    domain_rows = []
    missing_rows = []

    for (exp, domain), g in records.groupby(["experiment", "domain"], sort=True):
        row = {
            "experiment": exp,
            "domain": domain,
            "n_rows": int(len(g)),
            "n_unique_dates": int(g["date_parsed"].nunique()),
            "first_date": g["date_parsed"].min().strftime("%Y-%m-%d"),
            "last_date": g["date_parsed"].max().strftime("%Y-%m-%d"),
            "n_unique_times": int(g["measurement_time"].fillna("").nunique()),
            "times": ",".join(sorted(set(g["measurement_time"].dropna().astype(str)))),
        }

        for var in ENV_VARS:
            numeric = pd.to_numeric(g[var], errors="coerce")
            n = int(numeric.notna().sum())
            missing = int(numeric.isna().sum())
            row[f"{var}_n"] = n
            row[f"{var}_missing"] = missing
            row[f"{var}_missing_pct"] = 100.0 * missing / len(g) if len(g) else np.nan

            missing_rows.append({
                "experiment": exp,
                "domain": domain,
                "variable": var,
                "n_rows": int(len(g)),
                "n_observed": n,
                "n_missing": missing,
                "missing_pct": 100.0 * missing / len(g) if len(g) else np.nan,
            })

        domain_rows.append(row)

    return pd.DataFrame(domain_rows), pd.DataFrame(missing_rows)


def harvest_date_candidates(df: pd.DataFrame, experiment: str, sheet: str) -> list[dict[str, Any]]:
    """
    Conservative scan for explicit date-like cells in a harvest sheet.
    Does not infer a harvest date from workbook name or experiment number.
    """
    candidates = []
    date_word_cells = []

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            txt = clean(df.iat[r, c])
            if re.search(r"\bdate\b", txt, flags=re.I):
                date_word_cells.append((r, c, txt))

    # Search cells near explicit "date" labels first.
    inspected = set()
    for r, c, label in date_word_cells:
        for rr in range(max(0, r - 2), min(df.shape[0], r + 3)):
            for cc in range(max(0, c - 2), min(df.shape[1], c + 4)):
                inspected.add((rr, cc))
                raw = df.iat[rr, cc]
                d = parse_date_dayfirst(raw)
                if pd.notna(d):
                    candidates.append({
                        "experiment": experiment,
                        "sheet": sheet,
                        "label_row": r,
                        "label_col": c,
                        "label": label,
                        "candidate_row": rr,
                        "candidate_col": cc,
                        "raw_value": clean(raw),
                        "dayfirst_date": d.strftime("%Y-%m-%d"),
                        "basis": "near_explicit_date_label",
                    })

    # Also scan top 12 rows for actual datetime cells or obvious date strings.
    for r in range(min(12, df.shape[0])):
        for c in range(df.shape[1]):
            if (r, c) in inspected:
                continue
            raw = df.iat[r, c]
            txt = clean(raw)
            if not txt:
                continue

            looks_date = bool(
                re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", txt)
                or re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", txt)
            )
            is_datetime_obj = isinstance(raw, (pd.Timestamp,))
            if looks_date or is_datetime_obj:
                d = parse_date_dayfirst(raw)
                if pd.notna(d):
                    candidates.append({
                        "experiment": experiment,
                        "sheet": sheet,
                        "label_row": np.nan,
                        "label_col": np.nan,
                        "label": "",
                        "candidate_row": r,
                        "candidate_col": c,
                        "raw_value": txt,
                        "dayfirst_date": d.strftime("%Y-%m-%d"),
                        "basis": "top_rows_date_like_cell",
                    })

    # Exact de-duplication only.
    if not candidates:
        return []
    out = pd.DataFrame(candidates).drop_duplicates()
    return out.to_dict("records")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    archive = find_zip()
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Required P7.4 linkage file not found: {RECORDS_FILE}\n"
            "Run Phase 7.4 first."
        )

    records = pd.read_csv(RECORDS_FILE)
    required = {
        "experiment", "domain", "date", "measurement_time",
        "ph", "ec", "tds", "water_temperature"
    }
    missing_cols = sorted(required - set(records.columns))
    if missing_cols:
        raise RuntimeError(f"P7.4 records file missing columns: {missing_cols}")

    raw_date_rows = []
    harvest_candidates = []
    workbook_rows = []
    errors = []

    with zipfile.ZipFile(archive, "r") as zf:
        members = [
            n for n in zf.namelist()
            if Path(n).suffix.lower() in {".xlsx", ".xlsm"}
            and not Path(n).name.startswith("~$")
        ]

        for member in sorted(members):
            exp = experiment_from_name(member)
            if exp not in {"EXP1", "EXP2", "EXP3"}:
                continue

            try:
                raw_bytes = zf.read(member)
                bio = BytesIO(raw_bytes)
                xls = pd.ExcelFile(bio, engine="openpyxl")

                portable = find_sheet(xls.sheet_names, portable=True)
                harvest = find_sheet(xls.sheet_names, portable=False)

                workbook_rows.append({
                    "experiment": exp,
                    "workbook": Path(member).name,
                    "portable_sheet": portable or "",
                    "harvest_sheet": harvest or "",
                })

                if portable:
                    bio.seek(0)
                    portable_df = pd.read_excel(
                        bio, sheet_name=portable, header=None,
                        dtype=object, engine="openpyxl"
                    )
                    raw_date_rows.extend(raw_portable_date_audit(portable_df, exp))
                else:
                    errors.append(f"{exp}: portable sheet not found")

                if harvest:
                    bio.seek(0)
                    harvest_df = pd.read_excel(
                        bio, sheet_name=harvest, header=None,
                        dtype=object, engine="openpyxl"
                    )
                    harvest_candidates.extend(
                        harvest_date_candidates(harvest_df, exp, harvest)
                    )
                else:
                    errors.append(f"{exp}: harvest sheet not found")

            except Exception as exc:
                errors.append(
                    f"{exp} {Path(member).name}: {type(exc).__name__}: {exc}"
                )

    raw_dates = pd.DataFrame(raw_date_rows)
    sequence = infer_sequence_quality(raw_dates) if not raw_dates.empty else pd.DataFrame()
    domain_audit, missingness = linkage_record_audit(records)
    harvest_df = pd.DataFrame(harvest_candidates)
    workbook_df = pd.DataFrame(workbook_rows)

    # Temporal relation to harvest is evaluated only if there is one unambiguous
    # explicit harvest-date candidate per experiment.
    harvest_resolution = []
    for exp in ("EXP1", "EXP2", "EXP3"):
        if harvest_df.empty:
            dates = []
        else:
            dates = sorted(
                harvest_df.loc[
                    harvest_df["experiment"] == exp, "dayfirst_date"
                ].dropna().unique().tolist()
            )

        harvest_resolution.append({
            "experiment": exp,
            "candidate_dates": ",".join(dates),
            "n_unique_candidate_dates": len(dates),
            "explicit_harvest_date_resolved": len(dates) == 1,
            "resolved_harvest_date": dates[0] if len(dates) == 1 else "",
        })

    harvest_resolution_df = pd.DataFrame(harvest_resolution)

    temporal_relation_rows = []
    for r in domain_audit.itertuples(index=False):
        hr = harvest_resolution_df[
            harvest_resolution_df["experiment"] == r.experiment
        ].iloc[0]

        if bool(hr["explicit_harvest_date_resolved"]):
            harvest_date = pd.Timestamp(hr["resolved_harvest_date"])
            first = pd.Timestamp(r.first_date)
            last = pd.Timestamp(r.last_date)
            temporal_relation_rows.append({
                "experiment": r.experiment,
                "domain": r.domain,
                "first_environment_date": r.first_date,
                "last_environment_date": r.last_date,
                "harvest_date": harvest_date.strftime("%Y-%m-%d"),
                "all_environment_on_or_before_harvest": bool(last <= harvest_date),
                "days_last_environment_to_harvest": int((harvest_date - last).days),
                "environment_window_days": int((last - first).days),
            })
        else:
            temporal_relation_rows.append({
                "experiment": r.experiment,
                "domain": r.domain,
                "first_environment_date": r.first_date,
                "last_environment_date": r.last_date,
                "harvest_date": "",
                "all_environment_on_or_before_harvest": "",
                "days_last_environment_to_harvest": "",
                "environment_window_days": int(
                    (pd.Timestamp(r.last_date) - pd.Timestamp(r.first_date)).days
                ),
            })

    temporal_relation = pd.DataFrame(temporal_relation_rows)

    # Conservative verdict.
    domain_counts = (
        domain_audit.groupby("experiment")["domain"].nunique().to_dict()
        if not domain_audit.empty else {}
    )
    eighteen_domains_ok = all(
        domain_counts.get(exp, 0) == 6 for exp in ("EXP1", "EXP2", "EXP3")
    )

    no_parse_errors = len(errors) == 0
    sequence_available = not sequence.empty
    any_negative_source_steps = (
        int(sequence["negative_steps_in_source_order"].sum()) > 0
        if sequence_available else True
    )

    # A temporal audit can pass the parser/integrity layer while still requiring
    # manual scientific review of ambiguous raw dates. We therefore distinguish
    # PASS from REVIEW REQUIRED.
    raw_dates_present = len(raw_dates) > 0
    base_integrity_pass = (
        eighteen_domains_ok
        and no_parse_errors
        and raw_dates_present
    )

    report = []
    report.append("PHASE 7.5 — TEMPORAL INTEGRITY AUDIT")
    report.append("=" * 96)
    report.append(f"Archive: {archive}")
    report.append(f"Archive SHA256: {sha256_file(archive)}")
    report.append("No model training. No augmentation. No Phase-7 design freeze.")
    report.append("")

    report.append("WORKBOOK DISCOVERY")
    report.append("-" * 96)
    for r in workbook_df.itertuples(index=False):
        report.append(
            f"{r.experiment}: portable={r.portable_sheet!r} | harvest={r.harvest_sheet!r}"
        )
    report.append("")

    report.append("RAW PORTABLE DATE AUDIT")
    report.append("-" * 96)
    report.append(f"Raw parseable date rows found: {len(raw_dates)}")
    if not raw_dates.empty:
        report.append(
            f"Ambiguous numeric date strings: "
            f"{int(raw_dates['ambiguous_numeric_date'].sum())}"
        )
        report.append(
            f"Rows where day-first and month-first interpretations differ: "
            f"{int(raw_dates['interpretations_differ'].sum())}"
        )
    for r in sequence.itertuples(index=False):
        report.append(
            f"{r.experiment}: rows={r.n_raw_date_rows}, "
            f"dayfirst_window={r.dayfirst_first}..{r.dayfirst_last}, "
            f"negative_source_steps={r.negative_steps_in_source_order}, "
            f"max_forward_gap_days={r.max_forward_gap_days}, "
            f"ambiguous={r.n_ambiguous_numeric}"
        )
    report.append("")

    report.append("DOMAIN TEMPORAL COVERAGE")
    report.append("-" * 96)
    report.append(f"Experiment-domain cells: {len(domain_audit)}")
    report.append(f"Domain counts: {domain_counts}")
    for r in domain_audit.itertuples(index=False):
        report.append(
            f"{r.experiment} {r.domain}: rows={r.n_rows}, dates={r.n_unique_dates}, "
            f"window={r.first_date}..{r.last_date}, times=[{r.times}]"
        )
    report.append("")

    report.append("MISSINGNESS")
    report.append("-" * 96)
    for r in missingness.itertuples(index=False):
        report.append(
            f"{r.experiment} {r.domain} {r.variable}: "
            f"observed={r.n_observed}/{r.n_rows}, "
            f"missing={r.n_missing} ({r.missing_pct:.2f}%)"
        )
    report.append("")

    report.append("HARVEST DATE DISCOVERY")
    report.append("-" * 96)
    if harvest_df.empty:
        report.append("No explicit harvest-date candidate was conservatively recovered.")
    else:
        for r in harvest_resolution_df.itertuples(index=False):
            report.append(
                f"{r.experiment}: candidates=[{r.candidate_dates}] | "
                f"resolved={r.explicit_harvest_date_resolved} | "
                f"harvest_date={r.resolved_harvest_date or 'UNRESOLVED'}"
            )
    report.append("")

    report.append("ENVIRONMENT -> HARVEST TEMPORAL RELATION")
    report.append("-" * 96)
    for r in temporal_relation.itertuples(index=False):
        report.append(
            f"{r.experiment} {r.domain}: env={r.first_environment_date}.."
            f"{r.last_environment_date} | harvest={r.harvest_date or 'UNRESOLVED'} | "
            f"on_or_before_harvest={r.all_environment_on_or_before_harvest}"
        )
    report.append("")

    report.append("DATA-QUALITY FLAGS")
    report.append("-" * 96)
    report.append(f"Errors: {len(errors)}")
    for e in errors:
        report.append(f"  - {e}")
    report.append(
        f"Negative chronological steps in raw source-row order: "
        f"{int(sequence['negative_steps_in_source_order'].sum()) if not sequence.empty else 'UNKNOWN'}"
    )
    report.append("")

    report.append("FINAL VERDICT")
    report.append("-" * 96)

    if not base_integrity_pass:
        verdict = "FAIL"
        report.append("FAIL")
        report.append(
            "The temporal audit did not reconstruct enough structure for Phase-7 design review."
        )
    elif any_negative_source_steps or (
        not raw_dates.empty and int(raw_dates["interpretations_differ"].sum()) > 0
    ):
        verdict = "REVIEW REQUIRED"
        report.append("REVIEW REQUIRED")
        report.append(
            "Domain linkage remains intact, but raw date interpretation/order requires "
            "review before environmental exposure windows or shift severity are frozen."
        )
    else:
        verdict = "PASS"
        report.append("PASS")
        report.append(
            "Portable date ordering and domain temporal coverage passed the structural audit."
        )

    unresolved_harvest = int(
        (~harvest_resolution_df["explicit_harvest_date_resolved"]).sum()
    )
    report.append(
        f"Explicit harvest dates unresolved for {unresolved_harvest}/3 experiments."
    )
    if unresolved_harvest:
        report.append(
            "This is not silently imputed. Harvest timing must be resolved before "
            "final exposure-window construction if harvest-relative features are used."
        )

    report.append("")
    report.append("RESEARCH LOCK STATUS")
    report.append("-" * 96)
    report.append("Primary target frozen: NO")
    report.append("Feature set frozen: NO")
    report.append("Domain definition frozen: NO")
    report.append("Shift family frozen: NO")
    report.append("Shift severity frozen: NO")
    report.append("Augmentation frozen: NO")
    report.append("Model training performed: NO")

    # Write outputs.
    raw_dates.to_csv(OUT / "phase7_raw_portable_dates.csv", index=False)
    sequence.to_csv(OUT / "phase7_date_sequence_diagnostics.csv", index=False)
    domain_audit.to_csv(OUT / "phase7_domain_temporal_coverage.csv", index=False)
    missingness.to_csv(OUT / "phase7_environment_missingness.csv", index=False)
    harvest_df.to_csv(OUT / "phase7_harvest_date_candidates.csv", index=False)
    harvest_resolution_df.to_csv(
        OUT / "phase7_harvest_date_resolution.csv", index=False
    )
    temporal_relation.to_csv(
        OUT / "phase7_environment_harvest_temporal_relation.csv", index=False
    )
    workbook_df.to_csv(OUT / "phase7_temporal_workbook_inventory.csv", index=False)

    report_path = OUT / "phase7_temporal_integrity_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "phase": "7.5",
        "archive_sha256": sha256_file(archive),
        "verdict": verdict,
        "domain_cells": int(len(domain_audit)),
        "domain_counts": {k: int(v) for k, v in domain_counts.items()},
        "raw_parseable_date_rows": int(len(raw_dates)),
        "ambiguous_numeric_dates": (
            int(raw_dates["ambiguous_numeric_date"].sum())
            if not raw_dates.empty else 0
        ),
        "differing_dayfirst_monthfirst_interpretations": (
            int(raw_dates["interpretations_differ"].sum())
            if not raw_dates.empty else 0
        ),
        "explicit_harvest_dates_unresolved": unresolved_harvest,
        "errors": errors,
        "model_training_performed": False,
        "phase7_design_frozen": False,
    }
    (OUT / "phase7_temporal_integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n".join(report))
    print("")
    print(f"Report written to: {report_path}")

    # REVIEW REQUIRED is a successful diagnostic execution, so return 0.
    # Only structural FAIL returns non-zero.
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
