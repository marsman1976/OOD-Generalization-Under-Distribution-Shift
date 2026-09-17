"""
Phase 7.4b — Portable Sheet Layout Inspection

Diagnostic only. Inspects the three "Water quality parametersPortabl"
worksheets in the Phase 7 archive with header=None so their real multi-row
layout can be reconstructed before changing the environmental linkage parser.

NO model training. NO data mutation. NO research variables are frozen.
"""

from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "phase7" / "raw"
ZIP_FILE = RAW_DIR / "all_months_sensory_data.zip"
OUT = ROOT / "results" / "experiment_7" / "portable_inspection"
EXTRACT_DIR = OUT / "raw_excel"
REPORT_FILE = OUT / "phase7_portable_layout_report.txt"
CSV_FILE = OUT / "phase7_portable_nonempty_cells.csv"

OUT.mkdir(parents=True, exist_ok=True)


def clean(value):
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def detect_experiment(filename):
    name = filename.lower()
    for n in (1, 2, 3):
        if any(x in name for x in (f"exp.{n}", f"exp {n}", f"exp{n}", f"experiment {n}")):
            return f"EXP{n}"
    return "UNKNOWN"


def is_portable_sheet(name):
    s = clean(name).lower()
    return ("portabl" in s) or ("portable" in s)


def domain_like(text):
    s = clean(text).upper().replace("_", " ")
    patterns = [
        r"\bR\s*\d+\s*[- ]?\s*T\s*\d+\b",
        r"\bT\s*\d+\s*[- ]?\s*R\s*\d+\b",
        r"\bREPLICATE\s*\d+\b",
        r"\bTREATMENT\s*\d+\b",
        r"\bT\s*\d+\b",
        r"\bR\s*\d+\b",
    ]
    return any(re.search(p, s) for p in patterns)


def variable_like(text):
    s = clean(text).lower()
    tokens = ("ph", "ec", "tds", "lux", "light", "humidity", "rh", "co2", "temp", "temperature", "water")
    return any(t in s for t in tokens)


if not ZIP_FILE.exists():
    candidates = sorted(RAW_DIR.glob("*.zip"))
    if len(candidates) == 1:
        ZIP_FILE = candidates[0]
    else:
        raise FileNotFoundError("Could not uniquely locate the Phase 7 ZIP archive.")

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE, "r") as zf:
    zf.extractall(EXTRACT_DIR)

excel_files = sorted(EXTRACT_DIR.rglob("*.xlsx"))
if not excel_files:
    raise RuntimeError("No XLSX workbooks found in the Phase 7 archive.")

report = []
cell_records = []
portable_sheet_count = 0

report.append("PHASE 7.4b — PORTABLE SHEET LAYOUT INSPECTION")
report.append("=" * 120)
report.append("")
report.append("PURPOSE")
report.append("-" * 120)
report.append("Inspect portable water-quality worksheets exactly as stored, including multi-row headers and domain-like labels.")
report.append("No model training, data mutation, inferred domain mapping, or research-variable freezing is performed.")
report.append("")

for workbook_path in excel_files:
    experiment = detect_experiment(workbook_path.name)
    xls = pd.ExcelFile(workbook_path, engine="openpyxl")

    for sheet_name in xls.sheet_names:
        if not is_portable_sheet(sheet_name):
            continue

        portable_sheet_count += 1
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        report.append("")
        report.append("=" * 120)
        report.append(f"{experiment} | WORKBOOK: {workbook_path.name}")
        report.append(f"SHEET: {sheet_name}")
        report.append(f"SHAPE: {raw.shape}")
        report.append("=" * 120)
        report.append("")

        # Full non-empty-cell inventory. This is lossless for diagnostic purposes.
        report.append("NON-EMPTY CELLS BY ROW")
        report.append("-" * 120)

        for r in range(len(raw)):
            row_parts = []
            for c in range(raw.shape[1]):
                value = clean(raw.iat[r, c])
                if not value:
                    continue

                row_parts.append(f"C{c}={value!r}")
                cell_records.append({
                    "experiment": experiment,
                    "workbook": workbook_path.name,
                    "sheet": sheet_name,
                    "row": r,
                    "column": c,
                    "value": value,
                    "domain_like": domain_like(value),
                    "variable_like": variable_like(value),
                })

            if row_parts:
                report.append(f"R{r}: " + " | ".join(row_parts))
            else:
                report.append(f"R{r}: <EMPTY>")

        report.append("")
        report.append("DOMAIN-LIKE / TREATMENT-REPLICATE CELLS")
        report.append("-" * 120)
        found_domain = False
        for r in range(len(raw)):
            for c in range(raw.shape[1]):
                value = clean(raw.iat[r, c])
                if value and domain_like(value):
                    report.append(f"R{r} C{c}: {value!r}")
                    found_domain = True
        if not found_domain:
            report.append("NONE DETECTED")

        report.append("")
        report.append("ENVIRONMENT-VARIABLE-LIKE CELLS")
        report.append("-" * 120)
        found_variable = False
        for r in range(len(raw)):
            for c in range(raw.shape[1]):
                value = clean(raw.iat[r, c])
                if value and variable_like(value):
                    report.append(f"R{r} C{c}: {value!r}")
                    found_variable = True
        if not found_variable:
            report.append("NONE DETECTED")

        report.append("")
        report.append("COLUMN CONTEXT: FIRST 12 NON-EMPTY VALUES PER COLUMN")
        report.append("-" * 120)
        for c in range(raw.shape[1]):
            vals = []
            for r in range(len(raw)):
                value = clean(raw.iat[r, c])
                if value:
                    vals.append(f"R{r}:{value}")
                if len(vals) >= 12:
                    break
            if vals:
                report.append(f"C{c}: " + " || ".join(vals))

        report.append("")
        report.append("NUMERIC DENSITY BY COLUMN")
        report.append("-" * 120)
        for c in range(raw.shape[1]):
            series = raw.iloc[:, c]
            nonempty = series.map(clean).ne("")
            numeric = pd.to_numeric(series, errors="coerce")
            n_nonempty = int(nonempty.sum())
            n_numeric = int(numeric.notna().sum())
            if n_nonempty:
                report.append(
                    f"C{c}: nonempty={n_nonempty}, numeric={n_numeric}, "
                    f"numeric_fraction={n_numeric / n_nonempty:.3f}"
                )

report.append("")
report.append("=" * 120)
report.append("VALIDATION")
report.append("-" * 120)
report.append(f"Excel workbooks found: {len(excel_files)}")
report.append(f"Portable sheets inspected: {portable_sheet_count}")
report.append(f"Non-empty cells catalogued: {len(cell_records)}")
report.append("Expected portable sheets from P7.4: 3")
report.append("")

if portable_sheet_count == 3:
    report.append("P7.4b PORTABLE LAYOUT INSPECTION: PASS")
else:
    report.append("P7.4b PORTABLE LAYOUT INSPECTION: REVIEW REQUIRED")

report.append("")
report.append("RESEARCH LOCK STATUS")
report.append("-" * 120)
report.append("Primary target frozen: NO")
report.append("Feature set frozen: NO")
report.append("Domain definition frozen: NO")
report.append("Shift family frozen: NO")
report.append("Shift severity frozen: NO")
report.append("Augmentation frozen: NO")
report.append("Model training performed: NO")

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
pd.DataFrame(cell_records).to_csv(CSV_FILE, index=False)

print("=" * 88)
print("PHASE 7.4b — PORTABLE SHEET LAYOUT INSPECTION")
print("=" * 88)
print(f"Portable sheets inspected: {portable_sheet_count}")
print(f"Non-empty cells catalogued: {len(cell_records)}")
print(f"Report: {REPORT_FILE}")
print(f"Cell inventory: {CSV_FILE}")
print("")
if portable_sheet_count == 3:
    print("P7.4b PORTABLE LAYOUT INSPECTION: PASS")
else:
    print("P7.4b PORTABLE LAYOUT INSPECTION: REVIEW REQUIRED")
print("NO MODEL TRAINING WAS PERFORMED.")