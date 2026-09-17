"""
Phase 7.0 / 7.1 — Real Data Audit
=================================

Sheet-aware audit for HydroGrowNet real-data files.

IMPORTANT:
This script performs NO:
- model training
- augmentation
- feature engineering
- target selection
- domain selection
- shift construction

Expected raw archive:
data/phase7/raw/all_months_sensory_data.zip

Outputs:
results/experiment_7/audit/
"""

from pathlib import Path
import zipfile
import json
import hashlib
import shutil
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "phase7" / "raw"

EXPECTED_ZIP = (
    RAW_DIR / "all_months_sensory_data.zip"
)

OUT = (
    ROOT
    / "results"
    / "experiment_7"
    / "audit"
)

EXTRACT_DIR = (
    OUT
    / "extracted_for_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def sha256(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ------------------------------------------------------------
# Safe JSON conversion
# ------------------------------------------------------------

def json_safe(value):

    if isinstance(
        value,
        (np.integer,)
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,)
    ):

        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        (pd.Timestamp,)
    ):
        return str(value)

    return value


# ------------------------------------------------------------
# CSV reader
# ------------------------------------------------------------

def try_read_csv(path):

    attempts = [

        {
            "sep": ",",
            "encoding": "utf-8"
        },

        {
            "sep": ";",
            "encoding": "utf-8"
        },

        {
            "sep": "\t",
            "encoding": "utf-8"
        },

        {
            "sep": ",",
            "encoding": "latin1"
        },

        {
            "sep": ";",
            "encoding": "latin1"
        },
    ]

    best = None
    errors = []

    for kwargs in attempts:

        try:

            df = pd.read_csv(
                path,
                **kwargs
            )

            if len(df.columns) <= 1:
                continue

            if (
                best is None
                or len(df.columns)
                > len(best.columns)
            ):
                best = df

        except Exception as exc:

            errors.append(
                repr(exc)
            )

    return best, errors


# ------------------------------------------------------------
# Excel workbook reader
# ------------------------------------------------------------

def read_excel_workbook(path):

    """
    Read EVERY sheet in an Excel workbook.

    Returns:
        tables
        errors

    tables:
        list of dictionaries:
        {
            "sheet": sheet_name,
            "df": dataframe
        }
    """

    tables = []
    errors = []

    try:

        workbook = pd.ExcelFile(
            path,
            engine="openpyxl"
        )

    except Exception as exc:

        errors.append(
            {
                "sheet": None,
                "error": repr(exc)
            }
        )

        return tables, errors


    for sheet in workbook.sheet_names:

        try:

            df = pd.read_excel(
                workbook,
                sheet_name=sheet
            )

            tables.append(
                {
                    "sheet": str(sheet),
                    "df": df
                }
            )

        except Exception as exc:

            errors.append(
                {
                    "sheet": str(sheet),
                    "error": repr(exc)
                }
            )

    return tables, errors


# ------------------------------------------------------------
# Datetime detection
# ------------------------------------------------------------

def detect_datetime_columns(df):

    candidates = []

    for col in df.columns:

        name = str(col).lower()

        name_hint = any(
            token in name
            for token in [
                "time",
                "date",
                "timestamp",
                "datetime",
                "day",
            ]
        )

        if not name_hint:
            continue

        try:

            parsed = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            success = (
                parsed.notna().mean()
            )

            if success >= 0.50:

                candidates.append(
                    {
                        "column": str(col),
                        "parse_fraction": float(
                            success
                        ),
                        "min": (
                            str(parsed.min())
                            if parsed.notna().any()
                            else None
                        ),
                        "max": (
                            str(parsed.max())
                            if parsed.notna().any()
                            else None
                        ),
                    }
                )

        except Exception:
            pass

    return candidates


# ------------------------------------------------------------
# Numeric summary
# ------------------------------------------------------------

def numeric_summary(
    df,
    file_name,
    sheet_name
):

    numeric = df.select_dtypes(
        include=[np.number]
    )

    records = []

    for col in numeric.columns:

        s = df[col]

        valid = s.dropna()

        records.append(
            {
                "file": file_name,
                "sheet": sheet_name,
                "column": str(col),

                "count": int(
                    valid.count()
                ),

                "missing": int(
                    s.isna().sum()
                ),

                "missing_fraction": float(
                    s.isna().mean()
                ),

                "mean": (
                    float(valid.mean())
                    if len(valid)
                    else np.nan
                ),

                "std": (
                    float(valid.std())
                    if len(valid) > 1
                    else np.nan
                ),

                "min": (
                    float(valid.min())
                    if len(valid)
                    else np.nan
                ),

                "q01": (
                    float(
                        valid.quantile(.01)
                    )
                    if len(valid)
                    else np.nan
                ),

                "q25": (
                    float(
                        valid.quantile(.25)
                    )
                    if len(valid)
                    else np.nan
                ),

                "median": (
                    float(
                        valid.median()
                    )
                    if len(valid)
                    else np.nan
                ),

                "q75": (
                    float(
                        valid.quantile(.75)
                    )
                    if len(valid)
                    else np.nan
                ),

                "q99": (
                    float(
                        valid.quantile(.99)
                    )
                    if len(valid)
                    else np.nan
                ),

                "max": (
                    float(valid.max())
                    if len(valid)
                    else np.nan
                ),

                "n_unique": int(
                    valid.nunique()
                ),
            }
        )

    return records


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PHASE 7 REAL-DATA AUDIT — FIXED VERSION")
print("=" * 80)


# ============================================================
# LOCATE ZIP
# ============================================================

ZIP_FILE = EXPECTED_ZIP

if not ZIP_FILE.exists():

    zip_candidates = sorted(
        RAW_DIR.glob("*.zip")
    )

    if len(zip_candidates) == 1:

        ZIP_FILE = zip_candidates[0]

        print(
            "\nExpected filename not found."
        )

        print(
            f"Using detected archive: "
            f"{ZIP_FILE.name}"
        )

    else:

        raise FileNotFoundError(
            "\nCould not uniquely identify "
            "the Phase 7 ZIP archive.\n\n"
            f"Expected:\n{EXPECTED_ZIP}\n\n"
            f"ZIP candidates:\n"
            f"{[x.name for x in zip_candidates]}"
        )


print(
    f"\nArchive: {ZIP_FILE}"
)

print(
    f"Archive size: "
    f"{ZIP_FILE.stat().st_size:,} bytes"
)

archive_hash = sha256(
    ZIP_FILE
)

print(
    f"SHA256: {archive_hash}"
)


# ============================================================
# CLEAN OLD AUDIT EXTRACTION
# ============================================================

if EXTRACT_DIR.exists():

    shutil.rmtree(
        EXTRACT_DIR
    )

EXTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXTRACT ARCHIVE
# ============================================================

archive_records = []

with zipfile.ZipFile(
    ZIP_FILE,
    "r"
) as z:

    members = z.infolist()

    print(
        f"\nArchive members: "
        f"{len(members)}"
    )

    for member in members:

        archive_records.append(
            {
                "filename":
                    member.filename,

                "compressed_size":
                    member.compress_size,

                "uncompressed_size":
                    member.file_size,

                "is_directory":
                    member.is_dir(),
            }
        )

    z.extractall(
        EXTRACT_DIR
    )


pd.DataFrame(
    archive_records
).to_csv(
    OUT / "archive_inventory.csv",
    index=False
)


# ============================================================
# DISCOVER TABULAR FILES
# ============================================================

tabular_extensions = {
    ".csv",
    ".txt",
    ".tsv",
    ".xlsx",
    ".xls",
}

tabular_files = sorted(
    [
        p
        for p in EXTRACT_DIR.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in tabular_extensions
        )
    ]
)


print(
    f"Tabular files found: "
    f"{len(tabular_files)}"
)


if not tabular_files:

    raise RuntimeError(
        "PHASE 7 AUDIT FAILED: "
        "No supported tabular files "
        "were found in the archive."
    )


# ============================================================
# STORAGE
# ============================================================

file_inventory = []

table_inventory = []

column_inventory = []

missingness_records = []

numeric_records = []

datetime_records = []

correlation_records = []

read_error_records = []


# ============================================================
# PROCESS ONE TABLE
# ============================================================

def process_table(
    df,
    relative_file,
    sheet_name
):

    rows = len(df)
    columns = len(df.columns)

    duplicate_rows = int(
        df.duplicated().sum()
    )


    table_inventory.append(
        {
            "file": relative_file,
            "sheet": sheet_name,
            "read_success": True,
            "rows": int(rows),
            "columns": int(columns),
            "duplicate_rows":
                duplicate_rows,
        }
    )


    print(
        f"    Rows: {rows:,}"
    )

    print(
        f"    Columns: {columns}"
    )

    print(
        f"    Duplicate rows: "
        f"{duplicate_rows:,}"
    )


    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    for col in df.columns:

        s = df[col]

        column_inventory.append(
            {
                "file":
                    relative_file,

                "sheet":
                    sheet_name,

                "column":
                    str(col),

                "dtype":
                    str(s.dtype),

                "n_unique":
                    int(
                        s.nunique(
                            dropna=True
                        )
                    ),

                "missing":
                    int(
                        s.isna().sum()
                    ),

                "missing_fraction":
                    float(
                        s.isna().mean()
                    ),
            }
        )


        missingness_records.append(
            {
                "file":
                    relative_file,

                "sheet":
                    sheet_name,

                "column":
                    str(col),

                "missing":
                    int(
                        s.isna().sum()
                    ),

                "total":
                    int(len(s)),

                "missing_fraction":
                    float(
                        s.isna().mean()
                    ),
            }
        )


    # --------------------------------------------------------
    # Numeric summaries
    # --------------------------------------------------------

    numeric_records.extend(
        numeric_summary(
            df,
            relative_file,
            sheet_name
        )
    )


    # --------------------------------------------------------
    # Datetime candidates
    # --------------------------------------------------------

    datetime_candidates = (
        detect_datetime_columns(df)
    )

    for item in datetime_candidates:

        item["file"] = (
            relative_file
        )

        item["sheet"] = (
            sheet_name
        )

        datetime_records.append(
            item
        )


    # --------------------------------------------------------
    # Numeric correlations
    # --------------------------------------------------------

    numeric = df.select_dtypes(
        include=[np.number]
    )

    if len(numeric.columns) >= 2:

        corr = numeric.corr(
            method="pearson"
        )

        cols = corr.columns.tolist()

        for i in range(
            len(cols)
        ):

            for j in range(
                i + 1,
                len(cols)
            ):

                value = corr.loc[
                    cols[i],
                    cols[j]
                ]

                correlation_records.append(
                    {
                        "file":
                            relative_file,

                        "sheet":
                            sheet_name,

                        "variable_1":
                            str(cols[i]),

                        "variable_2":
                            str(cols[j]),

                        "pearson_r":
                            (
                                float(value)
                                if pd.notna(value)
                                else np.nan
                            ),

                        "abs_r":
                            (
                                float(abs(value))
                                if pd.notna(value)
                                else np.nan
                            ),
                    }
                )


# ============================================================
# READ ALL FILES
# ============================================================

for path in tabular_files:

    relative = str(
        path.relative_to(
            EXTRACT_DIR
        )
    )

    extension = (
        path.suffix.lower()
    )


    print()
    print("-" * 80)
    print(
        f"FILE: {relative}"
    )
    print("-" * 80)


    file_success = False
    n_sheets = 0
    n_tables_success = 0


    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------

    if extension in {
        ".xlsx",
        ".xls"
    }:

        try:

            tables, errors = (
                read_excel_workbook(
                    path
                )
            )


            n_sheets = (
                len(tables)
                + len(errors)
            )


            if tables:

                print(
                    f"  Readable sheets: "
                    f"{len(tables)}"
                )


            for table in tables:

                sheet = table[
                    "sheet"
                ]

                df = table[
                    "df"
                ]


                print(
                    f"\n  SHEET: {sheet}"
                )


                process_table(
                    df,
                    relative,
                    sheet
                )


                file_success = True

                n_tables_success += 1


            for error in errors:

                print(
                    "\n  READ ERROR:"
                )

                print(
                    f"    Sheet: "
                    f"{error['sheet']}"
                )

                print(
                    f"    Error: "
                    f"{error['error']}"
                )


                read_error_records.append(
                    {
                        "file":
                            relative,

                        "sheet":
                            error[
                                "sheet"
                            ],

                        "error":
                            error[
                                "error"
                            ],
                    }
                )


        except Exception as exc:

            error_text = (
                traceback.format_exc()
            )

            print(
                f"  FILE ERROR: "
                f"{repr(exc)}"
            )

            read_error_records.append(
                {
                    "file":
                        relative,

                    "sheet":
                        None,

                    "error":
                        error_text,
                }
            )


    # --------------------------------------------------------
    # CSV / TXT / TSV
    # --------------------------------------------------------

    else:

        df, errors = (
            try_read_csv(
                path
            )
        )


        if df is not None:

            process_table(
                df,
                relative,
                "__single_table__"
            )

            file_success = True

            n_tables_success = 1

            n_sheets = 1

        else:

            read_error_records.append(
                {
                    "file":
                        relative,

                    "sheet":
                        "__single_table__",

                    "error":
                        " | ".join(
                            errors
                        ),
                }
            )


    file_inventory.append(
        {
            "file":
                relative,

            "extension":
                extension,

            "read_success":
                file_success,

            "sheets_or_tables_detected":
                n_sheets,

            "tables_successfully_read":
                n_tables_success,
        }
    )


# ============================================================
# DATAFRAMES
# ============================================================

file_inventory_df = pd.DataFrame(
    file_inventory
)

tables_df = pd.DataFrame(
    table_inventory
)

columns_df = pd.DataFrame(
    column_inventory
)

missing_df = pd.DataFrame(
    missingness_records
)

numeric_df = pd.DataFrame(
    numeric_records
)

datetime_df = pd.DataFrame(
    datetime_records
)

correlation_df = pd.DataFrame(
    correlation_records
)

errors_df = pd.DataFrame(
    read_error_records
)


# ============================================================
# SAVE BASIC AUDIT TABLES
# ============================================================

file_inventory_df.to_csv(
    OUT / "file_inventory.csv",
    index=False
)

tables_df.to_csv(
    OUT / "table_inventory.csv",
    index=False
)

columns_df.to_csv(
    OUT / "column_inventory.csv",
    index=False
)

missing_df.to_csv(
    OUT / "missingness.csv",
    index=False
)

numeric_df.to_csv(
    OUT / "numeric_summary.csv",
    index=False
)

datetime_df.to_csv(
    OUT / "datetime_candidates.csv",
    index=False
)

correlation_df.to_csv(
    OUT / "numeric_correlations.csv",
    index=False
)

errors_df.to_csv(
    OUT / "read_errors.csv",
    index=False
)


# ============================================================
# CANDIDATE VARIABLE DISCOVERY
# ============================================================

candidate_patterns = {

    "air_temperature": [
        "air temperature",
        "air_temperature",
        "air temp",
        "temperature",
        "temp",
    ],

    "water_temperature": [
        "water temperature",
        "water_temperature",
        "water temp",
    ],

    "humidity": [
        "relative humidity",
        "humidity",
        "humid",
        "rh",
    ],

    "ph": [
        "ph",
    ],

    "ec": [
        "electrical conductivity",
        "conductivity",
        "ec",
    ],

    "light": [
        "light",
        "lux",
        "par",
        "radiation",
    ],

    "co2": [
        "co2",
        "carbon dioxide",
    ],

    "growth": [
        "growth",
        "height",
        "weight",
        "biomass",
        "yield",
        "leaf",
        "diameter",
        "area",
    ],

    "time": [
        "timestamp",
        "datetime",
        "date",
        "time",
        "day",
    ],
}


candidate_records = []


if not columns_df.empty:

    for _, row in (
        columns_df.iterrows()
    ):

        column = str(
            row["column"]
        )

        name = (
            column
            .lower()
            .strip()
        )


        for concept, patterns in (
            candidate_patterns.items()
        ):

            matched = [
                pattern
                for pattern in patterns
                if pattern in name
            ]


            if matched:

                candidate_records.append(
                    {
                        "file":
                            row["file"],

                        "sheet":
                            row["sheet"],

                        "column":
                            column,

                        "candidate_concept":
                            concept,

                        "matched_pattern":
                            ",".join(
                                matched
                            ),

                        "dtype":
                            row["dtype"],

                        "missing_fraction":
                            row[
                                "missing_fraction"
                            ],
                    }
                )


candidate_df = pd.DataFrame(
    candidate_records
)


candidate_df.to_csv(
    OUT / "candidate_variables.csv",
    index=False
)


# ============================================================
# STRONG CORRELATIONS
# ============================================================

if not correlation_df.empty:

    strong_corr = (
        correlation_df[
            correlation_df[
                "abs_r"
            ] >= 0.70
        ]
        .sort_values(
            "abs_r",
            ascending=False
        )
    )

else:

    strong_corr = (
        pd.DataFrame(
            columns=[
                "file",
                "sheet",
                "variable_1",
                "variable_2",
                "pearson_r",
                "abs_r",
            ]
        )
    )


strong_corr.to_csv(
    OUT / "strong_correlations.csv",
    index=False
)


# ============================================================
# HARD VALIDATION
# ============================================================

n_files_success = int(
    file_inventory_df[
        "read_success"
    ].sum()
)

n_tables_success = len(
    tables_df
)

n_columns = len(
    columns_df
)


audit_pass = (
    n_files_success > 0
    and n_tables_success > 0
    and n_columns > 0
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "phase":
        "PHASE7",

    "stage":
        "P7.0_P7.1_REAL_DATA_AUDIT",

    "audit_version":
        "2_sheet_aware",

    "source_archive":
        ZIP_FILE.name,

    "source_sha256":
        archive_hash,

    "source_size_bytes":
        ZIP_FILE.stat().st_size,

    "archive_members":
        len(members),

    "tabular_files_found":
        len(tabular_files),

    "files_read_successfully":
        n_files_success,

    "tables_or_sheets_read_successfully":
        n_tables_success,

    "columns_catalogued":
        n_columns,

    "read_errors":
        len(errors_df),

    "audit_pass":
        bool(audit_pass),

    "policy": {

        "training_performed":
            False,

        "augmentation_performed":
            False,

        "feature_engineering_performed":
            False,

        "target_selected":
            False,

        "domains_selected":
            False,

        "shift_definition_selected":
            False,

        "severity_definition_selected":
            False,

    },

    "important_note": (
        "Candidate variables are identified "
        "from column names only. "
        "They are not automatically accepted "
        "as Phase 7 research variables."
    ),
}


with open(
    OUT / "audit_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        default=json_safe
    )


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report = []

report.append(
    "PHASE 7 REAL-DATA AUDIT"
)

report.append(
    "=" * 78
)

report.append("")

report.append(
    f"Archive: "
    f"{ZIP_FILE.name}"
)

report.append(
    f"SHA256: "
    f"{archive_hash}"
)

report.append(
    f"Archive members: "
    f"{len(members)}"
)

report.append(
    f"Tabular files found: "
    f"{len(tabular_files)}"
)

report.append(
    f"Files successfully read: "
    f"{n_files_success}"
)

report.append(
    "Sheets/tables successfully read: "
    f"{n_tables_success}"
)

report.append(
    f"Columns catalogued: "
    f"{n_columns}"
)

report.append(
    f"Read errors: "
    f"{len(errors_df)}"
)

report.append(
    f"AUDIT PASS: "
    f"{audit_pass}"
)

report.append("")


# ------------------------------------------------------------
# File inventory
# ------------------------------------------------------------

report.append(
    "FILE INVENTORY"
)

report.append(
    "-" * 78
)

if not file_inventory_df.empty:

    report.append(
        file_inventory_df.to_string(
            index=False
        )
    )

else:

    report.append(
        "No file inventory available."
    )


report.append("")


# ------------------------------------------------------------
# Table inventory
# ------------------------------------------------------------

report.append(
    "TABLE / SHEET INVENTORY"
)

report.append(
    "-" * 78
)

if not tables_df.empty:

    report.append(
        tables_df.to_string(
            index=False
        )
    )

else:

    report.append(
        "No tables successfully read."
    )


report.append("")


# ------------------------------------------------------------
# Candidate variables
# ------------------------------------------------------------

report.append(
    "CANDIDATE RESEARCH VARIABLES"
)

report.append(
    "-" * 78
)

if not candidate_df.empty:

    report.append(
        candidate_df.to_string(
            index=False
        )
    )

else:

    report.append(
        "No candidate variables "
        "identified by column name."
    )


report.append("")


# ------------------------------------------------------------
# Datetime
# ------------------------------------------------------------

report.append(
    "DATETIME CANDIDATES"
)

report.append(
    "-" * 78
)

if not datetime_df.empty:

    report.append(
        datetime_df.to_string(
            index=False
        )
    )

else:

    report.append(
        "No datetime candidates detected."
    )


report.append("")


# ------------------------------------------------------------
# Read errors
# ------------------------------------------------------------

report.append(
    "READ ERRORS"
)

report.append(
    "-" * 78
)

if not errors_df.empty:

    report.append(
        errors_df.to_string(
            index=False
        )
    )

else:

    report.append(
        "No read errors."
    )


report.append("")


# ------------------------------------------------------------
# Research lock
# ------------------------------------------------------------

report.append(
    "RESEARCH DESIGN STATUS"
)

report.append(
    "-" * 78
)

report.append(
    "Model training performed: NO"
)

report.append(
    "Augmentation performed: NO"
)

report.append(
    "Target frozen: NO"
)

report.append(
    "Training composition frozen: NO"
)

report.append(
    "Shift family frozen: NO"
)

report.append(
    "Shift severity frozen: NO"
)

report.append("")

report.append(
    "Candidate variables in this report "
    "are descriptive candidates only."
)


with open(
    OUT / "phase7_audit_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("PHASE 7 AUDIT SUMMARY")
print("=" * 80)

print(
    f"Archive members:           "
    f"{len(members)}"
)

print(
    f"Tabular files:             "
    f"{len(tabular_files)}"
)

print(
    f"Files successfully read:   "
    f"{n_files_success}"
)

print(
    f"Tables/sheets read:        "
    f"{n_tables_success}"
)

print(
    f"Columns catalogued:        "
    f"{n_columns}"
)

print(
    f"Numeric variables:         "
    f"{len(numeric_df)}"
)

print(
    f"Datetime candidates:       "
    f"{len(datetime_df)}"
)

print(
    f"Candidate variables:       "
    f"{len(candidate_df)}"
)

print(
    f"Read errors:               "
    f"{len(errors_df)}"
)

print()


# ============================================================
# DO NOT ALLOW FALSE PASS
# ============================================================

if not audit_pass:

    print(
        "PHASE 7 REAL-DATA AUDIT: FAILED"
    )

    print()

    print(
        "The audit did not successfully "
        "read usable tables."
    )

    print(
        f"Inspect:\n"
        f"{OUT / 'read_errors.csv'}"
    )

    raise RuntimeError(
        "Phase 7 audit failed hard validation."
    )


print(
    "PHASE 7 REAL-DATA AUDIT: PASS"
)

print()

print(
    f"Outputs:\n{OUT}"
)

print()

print(
    "NO MODEL TRAINING WAS PERFORMED."
)

print(
    "NO TARGET, DOMAIN, SHIFT, OR "
    "SEVERITY DEFINITION HAS BEEN FROZEN."
)