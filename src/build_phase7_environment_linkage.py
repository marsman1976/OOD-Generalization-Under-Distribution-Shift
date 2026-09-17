"""
Phase 7.4 — Environmental History / Domain Linkage Audit
=========================================================

Purpose
-------
Reconstruct environmental-history tables from the HydroGrowNet Excel
workbooks and determine what can be linked safely to the 18 plant-level
harvest domains reconstructed in P7.3.

IMPORTANT:
This is an AUDIT/LINKAGE step only.

This script DOES NOT:
- train any ML model
- augment data
- freeze the target
- freeze the feature set
- freeze shift families or severity
- replicate hourly sensor rows into plant-level observations
- infer unsupported treatment/replicate mappings
- modify source files

Inputs
------
data/phase7/raw/all_months_sensory_data.zip

Required P7.3 output
--------------------
results/experiment_7/domain_inventory/harvest_plant_inventory.csv

Outputs
-------
results/experiment_7/environment_linkage/
    workbook_sheet_inventory.csv
    sensor_timeseries_long.csv
    sensor_environment_summary.csv
    portable_measurements_long.csv
    portable_environment_summary.csv
    portable_domain_measurements.csv
    portable_domain_summary.csv
    harvest_domain_inventory.csv
    domain_linkage_matrix.csv
    candidate_environment_features.csv
    unresolved_linkage_issues.csv
    phase7_environment_linkage_report.txt
    environment_linkage_manifest.json
    raw_excel/

Design principle
----------------
The effective outcome unit is a PLANT. Environmental measurements are
exposure-history information. Hourly sensor records are therefore not
treated as independent plant outcomes.
"""

from pathlib import Path
import zipfile
import shutil
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "phase7" / "raw"
ZIP_FILE = RAW_DIR / "all_months_sensory_data.zip"

P7_3_DIR = ROOT / "results" / "experiment_7" / "domain_inventory"
HARVEST_FILE = P7_3_DIR / "harvest_plant_inventory.csv"

OUT = ROOT / "results" / "experiment_7" / "environment_linkage"
EXTRACT_DIR = OUT / "raw_excel"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(value):
    text = clean_text(value).lower()

    replacements = {
        "°": "",
        "º": "",
        "%": " percent ",
        "&": " and ",
        "/": " ",
        "\\": " ",
        ".": " ",
        ",": " ",
        ":": " ",
        ";": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "-": " ",
        "_": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_experiment(filename):
    name = filename.lower()

    patterns = {
        "EXP1": [
            "exp.1",
            "exp 1",
            "exp1",
            "experiment 1",
            "experiment1",
        ],
        "EXP2": [
            "exp.2",
            "exp 2",
            "exp2",
            "experiment 2",
            "experiment2",
        ],
        "EXP3": [
            "exp.3",
            "exp 3",
            "exp3",
            "experiment 3",
            "experiment3",
        ],
    }

    for exp, terms in patterns.items():
        if any(term in name for term in terms):
            return exp

    return "UNKNOWN"


def safe_slug(value):
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text if text else "unnamed"


def numeric_or_nan(value):
    if pd.isna(value):
        return np.nan

    text = clean_text(value)

    if not text:
        return np.nan

    text_upper = text.upper()

    if text_upper in {
        "-",
        "--",
        "NA",
        "N/A",
        "NAN",
        "NONE",
    }:
        return np.nan

    # Decimal comma support when there is no decimal point.
    if "," in text and "." not in text:
        text = text.replace(",", ".")

    # Remove harmless unit-like spacing.
    text = text.strip()

    try:
        return float(text)
    except (ValueError, TypeError):
        return np.nan


def parse_datetime_series(series):
    """
    Conservative datetime parser.
    Returns parsed values plus success fraction among non-empty entries.
    """

    text = series.map(clean_text)
    nonempty = text.ne("")

    if nonempty.sum() == 0:
        return pd.Series(pd.NaT, index=series.index), 0.0

    parsed = pd.to_datetime(
        text.where(nonempty),
        errors="coerce",
        dayfirst=True,
    )

    fraction = float(
        parsed[nonempty].notna().mean()
    )

    return parsed, fraction


# ============================================================
# VARIABLE STANDARDIZATION
# ============================================================

def standardize_environment_variable(value):
    """
    Map raw header text to a conservative environmental variable name.
    Unknown columns return None.
    """

    name = normalize_text(value)

    if not name:
        return None

    # Date/time columns are handled separately.
    if name in {
        "date",
        "time",
        "tme",
        "datetime",
        "date time",
        "timestamp",
    }:
        return None

    # Relative humidity.
    if (
        name == "rh"
        or name.startswith("rh ")
        or "relative humidity" in name
        or "humidity" in name
    ):
        return "humidity_pct"

    # CO2.
    if (
        "co2" in name
        or "carbon dioxide" in name
    ):
        return "co2_ppm"

    # Electrical conductivity.
    if (
        name == "ec"
        or name.startswith("ec ")
        or "electrical conductivity" in name
    ):
        return "ec"

    # pH.
    if (
        name == "ph"
        or name.startswith("ph ")
        or "ph value" in name
    ):
        return "ph"

    # TDS.
    if (
        name == "tds"
        or name.startswith("tds ")
        or "total dissolved solids" in name
    ):
        return "tds"

    # Light / Lux.
    if (
        "lux" in name
        or name == "light"
        or name.startswith("light ")
        or "light intensity" in name
    ):
        return "light_lux"

    # Water temperature must be checked before generic temperature.
    if (
        ("water" in name and "temp" in name)
        or ("solution" in name and "temp" in name)
        or ("nutrient" in name and "temp" in name)
    ):
        return "water_temperature_c"

    # Air / ambient temperature.
    if (
        ("air" in name and "temp" in name)
        or ("ambient" in name and "temp" in name)
        or name.startswith("temperature")
        or name.startswith("temp")
    ):
        return "air_temperature_c"

    return None


# ============================================================
# DOMAIN PARSING
# ============================================================

def parse_domain_label(value):
    """
    Recognize domain identifiers such as:
      R1-T1
      R1T1
      Replicate 1 T1
      Replicate 1 Treatment 1
      T1 R1
      Treatment 1 Replicate 1
    """

    text = clean_text(value)

    if not text:
        return None

    upper = text.upper()
    upper = upper.replace("_", " ")
    upper = re.sub(r"\s+", " ", upper)

    patterns = [
        # R1-T1 / R1T1 / R1 T1
        (
            r"\bR\s*(\d+)\s*[- ]?\s*T\s*(\d+)\b",
            "RT",
        ),

        # Replicate 1 T1
        (
            r"\bREPLICATE\s*(\d+).*?\bT\s*(\d+)\b",
            "RT",
        ),

        # Replicate 1 Treatment 1
        (
            r"\bREPLICATE\s*(\d+).*?\bTREATMENT\s*(\d+)\b",
            "RT",
        ),

        # T1 R1
        (
            r"\bT\s*(\d+)\s*[- ]?\s*R\s*(\d+)\b",
            "TR",
        ),

        # Treatment 1 Replicate 1
        (
            r"\bTREATMENT\s*(\d+).*?\bREPLICATE\s*(\d+)\b",
            "TR",
        ),
    ]

    for pattern, order in patterns:
        match = re.search(pattern, upper)

        if not match:
            continue

        if order == "RT":
            replicate = int(match.group(1))
            treatment = int(match.group(2))
        else:
            treatment = int(match.group(1))
            replicate = int(match.group(2))

        return {
            "replicate": replicate,
            "treatment": treatment,
            "domain": f"R{replicate}-T{treatment}",
        }

    return None


def find_domain_tokens_in_row(row):
    found = []

    for col_idx, value in enumerate(row.tolist()):
        parsed = parse_domain_label(value)

        if parsed is not None:
            found.append(
                {
                    "column_index": col_idx,
                    "raw_label": clean_text(value),
                    **parsed,
                }
            )

    return found


# ============================================================
# SHEET CLASSIFICATION
# ============================================================

def classify_sheet(sheet_name, raw):
    name = normalize_text(sheet_name)

    if "harvest" in name:
        return "harvest"

    if "seedling" in name:
        return "seedling"

    if (
        "portable" in name
        or "handheld" in name
    ):
        return "portable"

    if (
        "sensor" in name
        or "sensory" in name
        or "water quality" in name
    ):
        return "sensor"

    # Content-based fallback.
    preview = " ".join(
        normalize_text(v)
        for v in raw.iloc[:15].to_numpy().ravel()
        if clean_text(v)
    )

    if "plant no" in preview and "total weight" in preview:
        return "harvest"

    env_hits = sum(
        token in preview
        for token in [
            "co2",
            "humidity",
            "rh",
            "temperature",
            "temp",
            "ph",
            "ec",
            "lux",
            "tds",
        ]
    )

    domain_hits = 0

    for row_idx in range(min(30, len(raw))):
        domain_hits += len(
            find_domain_tokens_in_row(raw.iloc[row_idx])
        )

    if env_hits >= 2 and domain_hits >= 1:
        return "portable"

    if env_hits >= 2:
        return "sensor"

    return "other"


# ============================================================
# HEADER DETECTION
# ============================================================

def header_score(row):
    """
    Score a raw row as a possible environmental-data header.
    """

    score = 0
    seen_variables = set()

    for value in row.tolist():
        text = normalize_text(value)

        if not text:
            continue

        if text in {
            "date",
            "time",
            "tme",
            "datetime",
            "date time",
            "timestamp",
        }:
            score += 1

        variable = standardize_environment_variable(value)

        if variable is not None:
            seen_variables.add(variable)

    score += 2 * len(seen_variables)

    return score, seen_variables


def find_best_header_row(raw, max_rows=40):
    best = None

    for row_idx in range(min(max_rows, len(raw))):
        score, variables = header_score(raw.iloc[row_idx])

        candidate = {
            "row_index": row_idx,
            "score": score,
            "variables": sorted(variables),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def make_unique_names(names):
    seen = defaultdict(int)
    output = []

    for name in names:
        base = name if name else "unnamed"
        count = seen[base]

        if count == 0:
            output.append(base)
        else:
            output.append(f"{base}_{count}")

        seen[base] += 1

    return output


# ============================================================
# SENSOR TIMESERIES PARSER
# ============================================================

def parse_sensor_sheet(
    raw,
    experiment,
    workbook_name,
    sheet_name,
):
    """
    Parse experiment-level sensor time series.

    This parser intentionally does NOT assign sensor rows to treatment/
    replicate domains unless the sheet itself contains such labels.
    """

    header = find_best_header_row(raw)

    if header is None or header["score"] < 4:
        return pd.DataFrame(), {
            "status": "HEADER_NOT_FOUND",
            "header_row": np.nan,
            "variables": "",
        }

    header_row = header["row_index"]
    header_values = raw.iloc[header_row].tolist()

    raw_names = [
        clean_text(v)
        for v in header_values
    ]

    generic_names = [
        safe_slug(v)
        for v in header_values
    ]

    generic_names = make_unique_names(generic_names)

    # Determine recognized variable columns.
    variable_columns = {}

    for col_idx, raw_name in enumerate(raw_names):
        variable = standardize_environment_variable(raw_name)

        if variable is not None:
            variable_columns[col_idx] = variable

    # Date/time candidates.
    date_cols = []
    time_cols = []
    datetime_cols = []

    for col_idx, raw_name in enumerate(raw_names):
        norm = normalize_text(raw_name)

        if norm in {"datetime", "date time", "timestamp"}:
            datetime_cols.append(col_idx)
        elif norm == "date":
            date_cols.append(col_idx)
        elif norm in {"time", "tme"}:
            time_cols.append(col_idx)

    records = []

    for row_idx in range(header_row + 1, len(raw)):
        row = raw.iloc[row_idx]

        values = {}

        for col_idx, variable in variable_columns.items():
            values[variable] = numeric_or_nan(
                row.iloc[col_idx]
            )

        n_numeric = sum(
            pd.notna(v)
            for v in values.values()
        )

        if n_numeric == 0:
            continue

        timestamp = pd.NaT

        # Combined datetime column.
        for col_idx in datetime_cols:
            parsed, frac = parse_datetime_series(
                pd.Series([row.iloc[col_idx]])
            )

            if frac > 0:
                timestamp = parsed.iloc[0]
                break

        # Separate date and time columns.
        if pd.isna(timestamp) and date_cols:
            date_text = clean_text(
                row.iloc[date_cols[0]]
            )

            time_text = (
                clean_text(row.iloc[time_cols[0]])
                if time_cols
                else ""
            )

            combined = (
                f"{date_text} {time_text}".strip()
            )

            parsed = pd.to_datetime(
                combined,
                errors="coerce",
                dayfirst=True,
            )

            if not pd.isna(parsed):
                timestamp = parsed

        record = {
            "experiment": experiment,
            "workbook": workbook_name,
            "sheet": sheet_name,
            "source_row": row_idx,
            "timestamp": timestamp,
        }

        record.update(values)
        records.append(record)

    df = pd.DataFrame(records)

    diagnostics = {
        "status": "OK" if not df.empty else "NO_NUMERIC_ROWS",
        "header_row": header_row,
        "variables": "|".join(
            sorted(set(variable_columns.values()))
        ),
        "rows_parsed": int(len(df)),
    }

    return df, diagnostics


# ============================================================
# PORTABLE SHEET PARSER
# ============================================================

def build_column_context(raw, header_row, col_idx, lookback=8):
    """
    Collect nearby header/context text for a column. This is used only
    for AUDIT purposes and domain detection.
    """

    pieces = []

    start = max(0, header_row - lookback)

    for row_idx in range(start, header_row + 1):
        value = clean_text(
            raw.iloc[row_idx, col_idx]
        )

        if value:
            pieces.append(value)

    return " | ".join(pieces)


def parse_portable_sheet(
    raw,
    experiment,
    workbook_name,
    sheet_name,
):
    """
    Conservative parser for portable environmental measurements.

    Strategy:
    1. Detect candidate header rows.
    2. Build column context from nearby multi-row headers.
    3. Detect environmental variable columns.
    4. Detect explicit R/T domain labels in column context.
    5. Extract numeric measurements.
    6. Never invent a domain when the workbook does not provide one.
    """

    # Candidate header rows can occur more than once.
    header_candidates = []

    for row_idx in range(min(60, len(raw))):
        score, variables = header_score(raw.iloc[row_idx])

        if score >= 4:
            header_candidates.append(
                {
                    "row_index": row_idx,
                    "score": score,
                    "variables": variables,
                }
            )

    if not header_candidates:
        return pd.DataFrame(), {
            "status": "HEADER_NOT_FOUND",
            "header_rows": "",
            "rows_parsed": 0,
            "domain_rows_found": 0,
        }

    # Keep separated header blocks rather than near-duplicate rows.
    selected_headers = []

    for candidate in header_candidates:
        if (
            not selected_headers
            or candidate["row_index"]
            - selected_headers[-1]["row_index"]
            > 3
        ):
            selected_headers.append(candidate)
        elif candidate["score"] > selected_headers[-1]["score"]:
            selected_headers[-1] = candidate

    records = []
    explicit_domains_seen = set()

    for h_idx, header in enumerate(selected_headers):
        header_row = header["row_index"]

        next_header_row = (
            selected_headers[h_idx + 1]["row_index"]
            if h_idx + 1 < len(selected_headers)
            else len(raw)
        )

        raw_header = raw.iloc[header_row].tolist()

        # Column metadata.
        col_meta = []

        for col_idx, header_value in enumerate(raw_header):
            variable = standardize_environment_variable(
                header_value
            )

            if variable is None:
                continue

            context = build_column_context(
                raw,
                header_row,
                col_idx,
            )

            domain = parse_domain_label(context)

            if domain is not None:
                explicit_domains_seen.add(domain["domain"])

            col_meta.append(
                {
                    "column_index": col_idx,
                    "variable": variable,
                    "context": context,
                    "domain": (
                        domain["domain"]
                        if domain is not None
                        else None
                    ),
                    "replicate": (
                        domain["replicate"]
                        if domain is not None
                        else np.nan
                    ),
                    "treatment": (
                        domain["treatment"]
                        if domain is not None
                        else np.nan
                    ),
                }
            )

        if not col_meta:
            continue

        # Detect date/time columns from current header.
        date_cols = []
        time_cols = []
        datetime_cols = []

        for col_idx, value in enumerate(raw_header):
            norm = normalize_text(value)

            if norm in {"datetime", "date time", "timestamp"}:
                datetime_cols.append(col_idx)
            elif norm == "date":
                date_cols.append(col_idx)
            elif norm in {"time", "tme"}:
                time_cols.append(col_idx)

        # Stop before next header block.
        for row_idx in range(header_row + 1, next_header_row):
            row = raw.iloc[row_idx]

            timestamp = pd.NaT

            for col_idx in datetime_cols:
                parsed = pd.to_datetime(
                    clean_text(row.iloc[col_idx]),
                    errors="coerce",
                    dayfirst=True,
                )

                if not pd.isna(parsed):
                    timestamp = parsed
                    break

            if pd.isna(timestamp) and date_cols:
                date_text = clean_text(
                    row.iloc[date_cols[0]]
                )

                time_text = (
                    clean_text(row.iloc[time_cols[0]])
                    if time_cols
                    else ""
                )

                combined = (
                    f"{date_text} {time_text}".strip()
                )

                parsed = pd.to_datetime(
                    combined,
                    errors="coerce",
                    dayfirst=True,
                )

                if not pd.isna(parsed):
                    timestamp = parsed

            for meta in col_meta:
                value = numeric_or_nan(
                    row.iloc[meta["column_index"]]
                )

                if pd.isna(value):
                    continue

                records.append(
                    {
                        "experiment": experiment,
                        "workbook": workbook_name,
                        "sheet": sheet_name,
                        "header_row": header_row,
                        "source_row": row_idx,
                        "source_column": meta["column_index"],
                        "timestamp": timestamp,
                        "domain": meta["domain"],
                        "replicate": meta["replicate"],
                        "treatment": meta["treatment"],
                        "variable": meta["variable"],
                        "value": value,
                        "column_context": meta["context"],
                    }
                )

    df = pd.DataFrame(records)

    diagnostics = {
        "status": "OK" if not df.empty else "NO_NUMERIC_ROWS",
        "header_rows": "|".join(
            str(x["row_index"])
            for x in selected_headers
        ),
        "rows_parsed": int(len(df)),
        "domain_rows_found": int(
            len(explicit_domains_seen)
        ),
        "explicit_domains": "|".join(
            sorted(explicit_domains_seen)
        ),
    }

    return df, diagnostics


# ============================================================
# SUMMARIZATION
# ============================================================

def summarize_wide_environment(
    df,
    group_cols,
    source_type,
):
    """
    Summarize a wide environmental table into mean/std/min/max/count
    for recognized environmental variables.
    """

    if df.empty:
        return pd.DataFrame()

    env_vars = [
        col
        for col in [
            "air_temperature_c",
            "humidity_pct",
            "co2_ppm",
            "water_temperature_c",
            "ph",
            "ec",
            "tds",
            "light_lux",
        ]
        if col in df.columns
    ]

    if not env_vars:
        return pd.DataFrame()

    records = []

    grouped = df.groupby(
        group_cols,
        dropna=False,
    )

    for keys, subset in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))
        base["source_type"] = source_type

        timestamps = (
            pd.to_datetime(
                subset["timestamp"],
                errors="coerce",
            )
            if "timestamp" in subset.columns
            else pd.Series(dtype="datetime64[ns]")
        )

        base["n_source_rows"] = int(len(subset))
        base["timestamp_start"] = (
            timestamps.min()
            if len(timestamps) and timestamps.notna().any()
            else pd.NaT
        )
        base["timestamp_end"] = (
            timestamps.max()
            if len(timestamps) and timestamps.notna().any()
            else pd.NaT
        )

        for variable in env_vars:
            values = pd.to_numeric(
                subset[variable],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            base[f"{variable}__n"] = int(len(values))
            base[f"{variable}__mean"] = float(values.mean())
            base[f"{variable}__std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            )
            base[f"{variable}__min"] = float(values.min())
            base[f"{variable}__max"] = float(values.max())

        records.append(base)

    return pd.DataFrame(records)


def summarize_long_environment(
    df,
    group_cols,
    source_type,
):
    """
    Summarize long variable/value portable measurements.
    """

    if df.empty:
        return pd.DataFrame()

    records = []

    grouped = df.groupby(
        group_cols + ["variable"],
        dropna=False,
    )

    for keys, subset in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        group_values = keys[:-1]
        variable = keys[-1]

        base = dict(
            zip(group_cols, group_values)
        )

        values = pd.to_numeric(
            subset["value"],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        base.update(
            {
                "source_type": source_type,
                "variable": variable,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

        timestamps = pd.to_datetime(
            subset["timestamp"],
            errors="coerce",
        )

        base["timestamp_start"] = (
            timestamps.min()
            if timestamps.notna().any()
            else pd.NaT
        )
        base["timestamp_end"] = (
            timestamps.max()
            if timestamps.notna().any()
            else pd.NaT
        )

        records.append(base)

    return pd.DataFrame(records)


# ============================================================
# CANDIDATE FEATURE INVENTORY
# ============================================================

def build_candidate_feature_inventory(
    sensor_df,
    portable_df,
):
    records = []

    candidate_vars = [
        "air_temperature_c",
        "humidity_pct",
        "co2_ppm",
        "water_temperature_c",
        "ph",
        "ec",
        "tds",
        "light_lux",
    ]

    experiments = ["EXP1", "EXP2", "EXP3"]

    for experiment in experiments:
        for variable in candidate_vars:
            sensor_n = 0
            portable_n = 0
            portable_domain_n = 0

            if (
                not sensor_df.empty
                and "experiment" in sensor_df.columns
                and variable in sensor_df.columns
            ):
                sensor_n = int(
                    pd.to_numeric(
                        sensor_df.loc[
                            sensor_df["experiment"] == experiment,
                            variable,
                        ],
                        errors="coerce",
                    )
                    .notna()
                    .sum()
                )

            if not portable_df.empty:
                portable_n = int(
                    (
                        (portable_df["experiment"] == experiment)
                        & (portable_df["variable"] == variable)
                    ).sum()
                )

                portable_domain_n = int(
                    portable_df.loc[
                        (portable_df["experiment"] == experiment)
                        & (portable_df["variable"] == variable)
                        & portable_df["domain"].notna(),
                        "domain",
                    ]
                    .nunique()
                )

            records.append(
                {
                    "experiment": experiment,
                    "variable": variable,
                    "sensor_measurements": sensor_n,
                    "portable_measurements": portable_n,
                    "portable_domains_explicitly_linked": portable_domain_n,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# INPUT VALIDATION
# ============================================================

print("=" * 88)
print("PHASE 7.4 — ENVIRONMENTAL HISTORY / DOMAIN LINKAGE AUDIT")
print("=" * 88)

if not ZIP_FILE.exists():
    candidates = sorted(RAW_DIR.glob("*.zip"))

    if len(candidates) == 1:
        ZIP_FILE = candidates[0]
    else:
        raise FileNotFoundError(
            "Could not locate data/phase7/raw/all_months_sensory_data.zip"
        )

if not HARVEST_FILE.exists():
    raise FileNotFoundError(
        "Required P7.3 file not found:\n"
        f"{HARVEST_FILE}\n\n"
        "Run the final fixed P7.3 domain inventory first."
    )

harvest = pd.read_csv(HARVEST_FILE)

required_harvest_cols = {
    "experiment",
    "domain",
    "replicate",
    "treatment",
    "plant_no",
    "plant_key",
}

missing = required_harvest_cols - set(harvest.columns)

if missing:
    raise RuntimeError(
        f"P7.3 harvest inventory is missing columns: {sorted(missing)}"
    )

harvest_domains = (
    harvest[
        [
            "experiment",
            "domain",
            "replicate",
            "treatment",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "experiment",
            "treatment",
            "replicate",
        ]
    )
    .reset_index(drop=True)
)

harvest_domain_counts = (
    harvest
    .groupby(
        [
            "experiment",
            "domain",
            "replicate",
            "treatment",
        ],
        dropna=False,
    )
    .agg(
        n_plants=("plant_key", "nunique"),
    )
    .reset_index()
)


# ============================================================
# EXTRACT WORKBOOKS
# ============================================================

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    z.extractall(EXTRACT_DIR)

excel_files = sorted(EXTRACT_DIR.rglob("*.xlsx"))

if not excel_files:
    raise RuntimeError("No .xlsx files found in the Phase 7 archive.")


# ============================================================
# PARSE WORKBOOKS
# ============================================================

sheet_inventory_records = []
sensor_frames = []
portable_frames = []

for workbook_path in excel_files:
    experiment = detect_experiment(workbook_path.name)

    workbook = pd.ExcelFile(
        workbook_path,
        engine="openpyxl",
    )

    print()
    print("-" * 88)
    print(experiment, workbook_path.name)
    print("-" * 88)

    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
        )

        sheet_type = classify_sheet(
            sheet_name,
            raw,
        )

        print(
            f"{sheet_name:<45} "
            f"type={sheet_type:<10} "
            f"shape={raw.shape}"
        )

        record = {
            "experiment": experiment,
            "workbook": workbook_path.name,
            "sheet": sheet_name,
            "sheet_type": sheet_type,
            "raw_rows": int(raw.shape[0]),
            "raw_columns": int(raw.shape[1]),
            "parse_status": "NOT_PARSED",
            "parsed_rows": 0,
            "header_info": "",
            "explicit_domains": "",
        }

        if sheet_type == "sensor":
            parsed, diagnostics = parse_sensor_sheet(
                raw,
                experiment,
                workbook_path.name,
                sheet_name,
            )

            record["parse_status"] = diagnostics["status"]
            record["parsed_rows"] = diagnostics.get(
                "rows_parsed",
                0,
            )
            record["header_info"] = str(
                diagnostics.get(
                    "header_row",
                    "",
                )
            )
            record["explicit_domains"] = diagnostics.get(
                "variables",
                "",
            )

            if not parsed.empty:
                sensor_frames.append(parsed)

        elif sheet_type == "portable":
            parsed, diagnostics = parse_portable_sheet(
                raw,
                experiment,
                workbook_path.name,
                sheet_name,
            )

            record["parse_status"] = diagnostics["status"]
            record["parsed_rows"] = diagnostics.get(
                "rows_parsed",
                0,
            )
            record["header_info"] = diagnostics.get(
                "header_rows",
                "",
            )
            record["explicit_domains"] = diagnostics.get(
                "explicit_domains",
                "",
            )

            if not parsed.empty:
                portable_frames.append(parsed)

        sheet_inventory_records.append(record)


sheet_inventory = pd.DataFrame(
    sheet_inventory_records
)

sensor_df = (
    pd.concat(
        sensor_frames,
        ignore_index=True,
        sort=False,
    )
    if sensor_frames
    else pd.DataFrame()
)

portable_df = (
    pd.concat(
        portable_frames,
        ignore_index=True,
        sort=False,
    )
    if portable_frames
    else pd.DataFrame()
)


# ============================================================
# SENSOR SUMMARIES
# ============================================================

sensor_summary = summarize_wide_environment(
    sensor_df,
    group_cols=["experiment"],
    source_type="experiment_sensor_timeseries",
)


# ============================================================
# PORTABLE SUMMARIES
# ============================================================

portable_summary = summarize_long_environment(
    portable_df,
    group_cols=["experiment"],
    source_type="portable_all",
)

if not portable_df.empty:
    portable_domain_df = portable_df[
        portable_df["domain"].notna()
    ].copy()
else:
    portable_domain_df = pd.DataFrame()

portable_domain_summary = summarize_long_environment(
    portable_domain_df,
    group_cols=[
        "experiment",
        "domain",
        "replicate",
        "treatment",
    ],
    source_type="portable_explicit_domain",
)


# ============================================================
# LINKAGE MATRIX
# ============================================================

linkage = harvest_domain_counts.copy()

# Experiment-level sensor availability.
sensor_availability = {}

if not sensor_df.empty:
    for experiment, subset in sensor_df.groupby("experiment"):
        env_cols = [
            col
            for col in [
                "air_temperature_c",
                "humidity_pct",
                "co2_ppm",
                "water_temperature_c",
                "ph",
                "ec",
                "tds",
                "light_lux",
            ]
            if col in subset.columns
        ]

        sensor_availability[experiment] = {
            "sensor_rows": int(len(subset)),
            "sensor_variables": "|".join(
                sorted(
                    col
                    for col in env_cols
                    if pd.to_numeric(
                        subset[col],
                        errors="coerce",
                    ).notna().any()
                )
            ),
        }

linkage["experiment_sensor_rows"] = linkage[
    "experiment"
].map(
    lambda exp: sensor_availability.get(
        exp,
        {},
    ).get(
        "sensor_rows",
        0,
    )
)

linkage["experiment_sensor_variables"] = linkage[
    "experiment"
].map(
    lambda exp: sensor_availability.get(
        exp,
        {},
    ).get(
        "sensor_variables",
        "",
    )
)

# Explicit portable domain availability.
portable_domain_lookup = {}

if not portable_domain_df.empty:
    grouped = portable_domain_df.groupby(
        [
            "experiment",
            "domain",
        ],
        dropna=False,
    )

    for (experiment, domain), subset in grouped:
        portable_domain_lookup[
            (experiment, domain)
        ] = {
            "rows": int(len(subset)),
            "variables": "|".join(
                sorted(
                    subset["variable"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            ),
        }

linkage["portable_domain_measurements"] = linkage.apply(
    lambda row: portable_domain_lookup.get(
        (row["experiment"], row["domain"]),
        {},
    ).get(
        "rows",
        0,
    ),
    axis=1,
)

linkage["portable_domain_variables"] = linkage.apply(
    lambda row: portable_domain_lookup.get(
        (row["experiment"], row["domain"]),
        {},
    ).get(
        "variables",
        "",
    ),
    axis=1,
)

linkage["has_experiment_sensor_history"] = (
    linkage["experiment_sensor_rows"] > 0
)

linkage["has_explicit_portable_domain_history"] = (
    linkage["portable_domain_measurements"] > 0
)

# Conservative linkage classification.
def classify_linkage(row):
    if row["has_explicit_portable_domain_history"]:
        return "EXPLICIT_DOMAIN_LINK_AVAILABLE"

    if row["has_experiment_sensor_history"]:
        return "EXPERIMENT_LEVEL_ONLY"

    return "NO_ENVIRONMENT_LINK_FOUND"


linkage["linkage_status"] = linkage.apply(
    classify_linkage,
    axis=1,
)


# ============================================================
# CANDIDATE ENVIRONMENT FEATURE INVENTORY
# ============================================================

candidate_features = build_candidate_feature_inventory(
    sensor_df,
    portable_df,
)


# ============================================================
# UNRESOLVED ISSUES
# ============================================================

issues = []

expected_experiments = {
    "EXP1",
    "EXP2",
    "EXP3",
}

observed_harvest_experiments = set(
    harvest["experiment"].unique()
)

if not expected_experiments <= observed_harvest_experiments:
    issues.append(
        {
            "severity": "ERROR",
            "issue": "MISSING_HARVEST_EXPERIMENT",
            "details": (
                "P7.3 harvest inventory does not contain all "
                "EXP1/EXP2/EXP3."
            ),
        }
    )

if len(harvest_domains) != 18:
    issues.append(
        {
            "severity": "WARNING",
            "issue": "UNEXPECTED_HARVEST_DOMAIN_COUNT",
            "details": (
                f"Expected 18 experiment-domain cells from P7.3; "
                f"found {len(harvest_domains)}."
            ),
        }
    )

if sensor_df.empty:
    issues.append(
        {
            "severity": "WARNING",
            "issue": "NO_SENSOR_TIMESERIES_PARSED",
            "details": (
                "No experiment-level sensor time series were parsed."
            ),
        }
    )

if portable_df.empty:
    issues.append(
        {
            "severity": "WARNING",
            "issue": "NO_PORTABLE_MEASUREMENTS_PARSED",
            "details": (
                "No portable environmental measurements were parsed."
            ),
        }
    )

n_explicit_domain_links = int(
    linkage[
        "has_explicit_portable_domain_history"
    ].sum()
)

if n_explicit_domain_links < len(linkage):
    issues.append(
        {
            "severity": "INFO",
            "issue": "NOT_ALL_DOMAINS_HAVE_EXPLICIT_PORTABLE_LINKS",
            "details": (
                f"{n_explicit_domain_links}/{len(linkage)} "
                "harvest domains currently have an explicit "
                "portable-measurement domain link. Unlinked domains "
                "must not be assigned environmental values by guesswork."
            ),
        }
    )

# Experiment-level sensor histories cannot by themselves distinguish
# treatment/replicate domains.
for experiment in sorted(
    linkage["experiment"].unique()
):
    subset = linkage[
        linkage["experiment"] == experiment
    ]

    if (
        subset["has_experiment_sensor_history"].any()
        and subset["has_explicit_portable_domain_history"].sum() == 0
    ):
        issues.append(
            {
                "severity": "INFO",
                "issue": "SENSOR_HISTORY_IS_EXPERIMENT_LEVEL",
                "details": (
                    f"{experiment}: sensor time series exists, but no "
                    "explicit domain-specific portable linkage was "
                    "parsed. Do not copy the same experiment-level "
                    "history into treatment/replicate domains and call "
                    "that domain-specific exposure."
                ),
            }
        )

issues_df = pd.DataFrame(
    issues,
    columns=[
        "severity",
        "issue",
        "details",
    ],
)


# ============================================================
# SAVE TABLES
# ============================================================

sheet_inventory.to_csv(
    OUT / "workbook_sheet_inventory.csv",
    index=False,
)

sensor_df.to_csv(
    OUT / "sensor_timeseries_long.csv",
    index=False,
)

sensor_summary.to_csv(
    OUT / "sensor_environment_summary.csv",
    index=False,
)

portable_df.to_csv(
    OUT / "portable_measurements_long.csv",
    index=False,
)

portable_summary.to_csv(
    OUT / "portable_environment_summary.csv",
    index=False,
)

portable_domain_df.to_csv(
    OUT / "portable_domain_measurements.csv",
    index=False,
)

portable_domain_summary.to_csv(
    OUT / "portable_domain_summary.csv",
    index=False,
)

harvest_domains.to_csv(
    OUT / "harvest_domain_inventory.csv",
    index=False,
)

linkage.to_csv(
    OUT / "domain_linkage_matrix.csv",
    index=False,
)

candidate_features.to_csv(
    OUT / "candidate_environment_features.csv",
    index=False,
)

issues_df.to_csv(
    OUT / "unresolved_linkage_issues.csv",
    index=False,
)


# ============================================================
# VALIDATION COUNTS
# ============================================================

sensor_rows = int(len(sensor_df))
portable_rows = int(len(portable_df))
portable_domain_rows = int(len(portable_domain_df))

sensor_experiments = (
    sorted(sensor_df["experiment"].dropna().unique().tolist())
    if not sensor_df.empty
    else []
)

portable_experiments = (
    sorted(portable_df["experiment"].dropna().unique().tolist())
    if not portable_df.empty
    else []
)

explicit_linked_domains = int(
    linkage[
        "has_explicit_portable_domain_history"
    ].sum()
)

experiment_sensor_domains = int(
    linkage[
        "has_experiment_sensor_history"
    ].sum()
)

fatal_errors = int(
    (issues_df["severity"] == "ERROR").sum()
    if not issues_df.empty
    else 0
)


# ============================================================
# REPORT
# ============================================================

report = []

report.append(
    "PHASE 7.4 — ENVIRONMENTAL HISTORY / DOMAIN LINKAGE AUDIT"
)
report.append("=" * 100)
report.append("")

report.append("PURPOSE")
report.append("-" * 100)
report.append(
    "Audit environmental-history sources and determine which "
    "measurements can be linked safely to the 18 P7.3 harvest domains."
)
report.append(
    "No machine-learning model was trained and no research variable "
    "was frozen."
)
report.append("")

report.append("P7.3 HARVEST BASIS")
report.append("-" * 100)
report.append(
    f"Harvest plant rows: {len(harvest)}"
)
report.append(
    f"Unique plants: {harvest['plant_key'].nunique()}"
)
report.append(
    f"Experiment-domain cells: {len(harvest_domains)}"
)
report.append("")

report.append("ENVIRONMENTAL PARSING")
report.append("-" * 100)
report.append(
    f"Excel workbooks: {len(excel_files)}"
)
report.append(
    f"Worksheets inspected: {len(sheet_inventory)}"
)
report.append(
    f"Sensor rows parsed: {sensor_rows}"
)
report.append(
    f"Sensor experiments: {', '.join(sensor_experiments) if sensor_experiments else 'NONE'}"
)
report.append(
    f"Portable measurement rows parsed: {portable_rows}"
)
report.append(
    f"Portable experiments: {', '.join(portable_experiments) if portable_experiments else 'NONE'}"
)
report.append(
    f"Portable rows with explicit domain labels: {portable_domain_rows}"
)
report.append("")

report.append("DOMAIN LINKAGE")
report.append("-" * 100)
report.append(
    f"Harvest domains with experiment-level sensor history: "
    f"{experiment_sensor_domains}/{len(linkage)}"
)
report.append(
    f"Harvest domains with explicit portable domain history: "
    f"{explicit_linked_domains}/{len(linkage)}"
)
report.append("")
report.append(
    linkage[
        [
            "experiment",
            "domain",
            "replicate",
            "treatment",
            "n_plants",
            "experiment_sensor_rows",
            "portable_domain_measurements",
            "linkage_status",
        ]
    ].to_string(index=False)
)
report.append("")

report.append("CANDIDATE ENVIRONMENT VARIABLES")
report.append("-" * 100)
report.append(
    candidate_features.to_string(index=False)
)
report.append("")

report.append("WORKSHEET PARSE INVENTORY")
report.append("-" * 100)
report.append(
    sheet_inventory[
        [
            "experiment",
            "sheet",
            "sheet_type",
            "raw_rows",
            "raw_columns",
            "parse_status",
            "parsed_rows",
            "header_info",
            "explicit_domains",
        ]
    ].to_string(index=False)
)
report.append("")

report.append("UNRESOLVED LINKAGE ISSUES")
report.append("-" * 100)

if issues_df.empty:
    report.append("NONE")
else:
    report.append(
        issues_df.to_string(index=False)
    )

report.append("")

report.append("SCIENTIFIC GUARDRAILS")
report.append("-" * 100)
report.append(
    "1. Hourly sensor rows are exposure-history measurements, "
    "not independent plant observations."
)
report.append(
    "2. Experiment-level sensor histories are not automatically "
    "treatment/replicate-specific."
)
report.append(
    "3. Environmental values are linked to a harvest domain only "
    "when the workbook structure provides an explicit mapping."
)
report.append(
    "4. Missing domain links are retained as unresolved; they are "
    "not filled by guesswork."
)
report.append(
    "5. No target, feature set, domain definition, shift family, "
    "shift severity, augmentation, or ML model is frozen here."
)
report.append("")

report.append("RESEARCH LOCK STATUS")
report.append("-" * 100)
report.append("Primary target frozen: NO")
report.append("Feature set frozen: NO")
report.append("Domain definition frozen: NO")
report.append("Shift family frozen: NO")
report.append("Shift severity frozen: NO")
report.append("Augmentation frozen: NO")
report.append("Model training performed: NO")
report.append("")

report.append("VALIDATION")
report.append("-" * 100)
report.append(
    f"Fatal audit errors: {fatal_errors}"
)
report.append(
    f"P7.3 harvest experiments recovered: "
    f"{', '.join(sorted(observed_harvest_experiments))}"
)
report.append(
    f"P7.3 harvest domains recovered: {len(harvest_domains)}"
)

REPORT_FILE = (
    OUT / "phase7_environment_linkage_report.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(report))


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "phase": "PHASE7",
    "stage": "P7.4_ENVIRONMENT_LINKAGE_AUDIT",
    "source_archive": ZIP_FILE.name,
    "p7_3_harvest_file": str(
        HARVEST_FILE.relative_to(ROOT)
    ),
    "harvest_rows": int(len(harvest)),
    "unique_plants": int(
        harvest["plant_key"].nunique()
    ),
    "harvest_domain_cells": int(
        len(harvest_domains)
    ),
    "sensor_rows_parsed": sensor_rows,
    "portable_measurements_parsed": portable_rows,
    "portable_domain_rows": portable_domain_rows,
    "domains_with_experiment_sensor_history": (
        experiment_sensor_domains
    ),
    "domains_with_explicit_portable_history": (
        explicit_linked_domains
    ),
    "fatal_errors": fatal_errors,
    "model_training": False,
    "research_variables_frozen": False,
    "guardrails": {
        "hourly_rows_are_not_independent_plants": True,
        "no_inferred_domain_mapping": True,
        "no_augmentation": True,
        "no_model_training": True,
    },
}

with open(
    OUT / "environment_linkage_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
        default=str,
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 88)
print("PHASE 7.4 ENVIRONMENT LINKAGE SUMMARY")
print("=" * 88)
print(
    f"P7.3 harvest plants:                       "
    f"{harvest['plant_key'].nunique()}"
)
print(
    f"P7.3 experiment-domain cells:              "
    f"{len(harvest_domains)}"
)
print(
    f"Sensor rows parsed:                        "
    f"{sensor_rows}"
)
print(
    f"Portable measurements parsed:              "
    f"{portable_rows}"
)
print(
    f"Portable rows explicitly domain-linked:    "
    f"{portable_domain_rows}"
)
print(
    f"Domains with experiment sensor history:    "
    f"{experiment_sensor_domains}/{len(linkage)}"
)
print(
    f"Domains with explicit portable history:    "
    f"{explicit_linked_domains}/{len(linkage)}"
)
print(
    f"Fatal audit errors:                        "
    f"{fatal_errors}"
)
print()

if fatal_errors == 0 and len(harvest_domains) == 18:
    print("PHASE 7.4 ENVIRONMENT LINKAGE AUDIT: PASS")
else:
    print("PHASE 7.4 ENVIRONMENT LINKAGE AUDIT: REVIEW REQUIRED")

print()
print(f"Report:\n{REPORT_FILE}")
print()
print("NO MODEL TRAINING WAS PERFORMED.")
print("NO X/Y/D/S/V/A DEFINITIONS WERE FROZEN.")
