"""
Phase 7.2 — Raw Excel Schema Reconstruction
===========================================

Purpose:
Inspect the REAL spreadsheet structure before:
- selecting predictors
- selecting target
- cleaning data
- constructing domains
- defining shifts
- defining severity
- augmentation
- model training

The HydroGrowNet workbooks contain irregular / multi-row Excel
headers. Therefore this script deliberately reads sheets using:

    header=None

Outputs:
results/experiment_7/schema_inspection/

NO MODEL TRAINING IS PERFORMED.
NO RESEARCH VARIABLES ARE FROZEN.
"""

from pathlib import Path
import zipfile
import shutil
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data"
    / "phase7"
    / "raw"
)

EXPECTED_ZIP = (
    RAW_DIR
    / "all_months_sensory_data.zip"
)

OUT = (
    ROOT
    / "results"
    / "experiment_7"
    / "schema_inspection"
)

EXTRACT_DIR = (
    OUT
    / "raw_excel"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Number of raw rows shown from each sheet.
PREVIEW_ROWS = 25

# Number of non-empty values sampled per column.
COLUMN_SAMPLE_SIZE = 8


# ============================================================
# HELPERS
# ============================================================

def clean_display(value):
    """
    Convert cell value to readable text without changing
    the underlying source data.
    """

    if pd.isna(value):
        return ""

    value = str(value)

    value = value.replace(
        "\n",
        " "
    )

    value = value.replace(
        "\r",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def detect_experiment(filename):

    name = filename.lower()

    if "exp.1" in name or "exp 1" in name:
        return "EXP1"

    if "exp.2" in name or "exp 2" in name:
        return "EXP2"

    if "exp.3" in name or "exp 3" in name:
        return "EXP3"

    return "UNKNOWN"


def row_nonempty_count(row):

    return int(
        row.notna().sum()
    )


def looks_like_header_row(row):
    """
    Heuristic only.

    This does NOT automatically decide the real header.

    It identifies rows that contain substantial text
    and might be part of a multi-row spreadsheet header.
    """

    values = [
        clean_display(v)
        for v in row.tolist()
        if clean_display(v)
    ]

    if not values:
        return False

    text_values = 0

    for value in values:

        # Does it contain alphabetic characters?
        if re.search(
            r"[A-Za-z]",
            value
        ):
            text_values += 1

    fraction_text = (
        text_values
        / len(values)
    )

    return (
        len(values) >= 2
        and fraction_text >= 0.50
    )


def keyword_hits(value):

    text = clean_display(
        value
    ).lower()

    patterns = {

        "temperature": [
            "temperature",
            "temp",
        ],

        "humidity": [
            "humidity",
            "humid",
            "relative humidity",
            "rh",
        ],

        "water": [
            "water",
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
            "radiation",
            "par",
        ],

        "co2": [
            "co2",
            "carbon dioxide",
        ],

        "growth": [
            "growth",
            "diameter",
            "height",
            "biomass",
            "weight",
            "yield",
            "leaf",
            "leaves",
            "head",
            "harvest",
        ],

        "time": [
            "date",
            "time",
            "timestamp",
            "day",
        ],

        "nutrient": [
            "nutrient",
            "nitrate",
            "nitrogen",
            "phosphate",
            "phosphorus",
            "potassium",
        ],
    }

    hits = []

    for concept, words in patterns.items():

        if any(
            word in text
            for word in words
        ):
            hits.append(
                concept
            )

    return hits


# ============================================================
# LOCATE ZIP
# ============================================================

print("=" * 80)
print("PHASE 7.2 — RAW SCHEMA INSPECTION")
print("=" * 80)


ZIP_FILE = EXPECTED_ZIP


if not ZIP_FILE.exists():

    candidates = sorted(
        RAW_DIR.glob("*.zip")
    )

    if len(candidates) == 1:

        ZIP_FILE = candidates[0]

        print(
            f"\nUsing detected archive: "
            f"{ZIP_FILE.name}"
        )

    else:

        raise FileNotFoundError(
            "\nCould not uniquely locate Phase 7 archive.\n"
            f"Expected:\n{EXPECTED_ZIP}\n"
            f"Found:\n{[p.name for p in candidates]}"
        )


print(
    f"\nArchive: {ZIP_FILE.name}"
)


# ============================================================
# CLEAN TEMPORARY EXTRACTION
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
# EXTRACT
# ============================================================

with zipfile.ZipFile(
    ZIP_FILE,
    "r"
) as z:

    z.extractall(
        EXTRACT_DIR
    )


excel_files = sorted(
    EXTRACT_DIR.rglob("*.xlsx")
)


if not excel_files:

    raise RuntimeError(
        "No XLSX files found."
    )


print(
    f"Excel workbooks found: "
    f"{len(excel_files)}"
)


# ============================================================
# STORAGE
# ============================================================

sheet_inventory = []

row_structure_records = []

cell_records = []

keyword_records = []

column_sample_records = []

report = []


report.append(
    "PHASE 7.2 — RAW EXCEL SCHEMA INSPECTION"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "IMPORTANT:"
)

report.append(
    "All worksheets were read with header=None."
)

report.append(
    "No target, feature, domain, shift, severity, or "
    "augmentation definition has been selected."
)

report.append("")


# ============================================================
# PROCESS WORKBOOKS
# ============================================================

for workbook_path in excel_files:

    experiment = detect_experiment(
        workbook_path.name
    )

    print()
    print("=" * 80)
    print(
        f"{experiment}: "
        f"{workbook_path.name}"
    )
    print("=" * 80)


    workbook = pd.ExcelFile(
        workbook_path,
        engine="openpyxl"
    )


    report.append("")
    report.append("#" * 100)
    report.append(
        f"WORKBOOK: {workbook_path.name}"
    )
    report.append(
        f"EXPERIMENT: {experiment}"
    )
    report.append("#" * 100)


    for sheet_name in workbook.sheet_names:

        print(
            f"\n  Inspecting sheet: "
            f"{sheet_name}"
        )


        # ----------------------------------------------------
        # CRITICAL:
        # read WITHOUT assuming a header
        # ----------------------------------------------------

        raw = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None
        )


        rows = len(raw)
        cols = len(raw.columns)


        nonempty_cells = int(
            raw.notna().sum().sum()
        )


        sheet_inventory.append(
            {
                "experiment":
                    experiment,

                "workbook":
                    workbook_path.name,

                "sheet":
                    sheet_name,

                "rows":
                    rows,

                "columns":
                    cols,

                "nonempty_cells":
                    nonempty_cells,
            }
        )


        print(
            f"    Raw shape: "
            f"{rows} rows x {cols} columns"
        )


        # ====================================================
        # REPORT HEADER
        # ====================================================

        report.append("")
        report.append("-" * 100)

        report.append(
            f"SHEET: {sheet_name}"
        )

        report.append(
            f"RAW SHAPE: "
            f"{rows} x {cols}"
        )

        report.append("-" * 100)

        report.append("")


        # ====================================================
        # RAW PREVIEW
        # ====================================================

        preview_n = min(
            PREVIEW_ROWS,
            rows
        )


        preview = raw.iloc[
            :preview_n
        ].copy()


        preview = preview.apply(
            lambda col: col.map(
                clean_display
            )
        )


        # Rename raw columns C000, C001...
        preview.columns = [
            f"C{i:03d}"
            for i in range(
                len(preview.columns)
            )
        ]


        report.append(
            f"FIRST {preview_n} RAW ROWS:"
        )

        report.append("")

        report.append(
            preview.to_string(
                index=True,
                max_cols=None
            )
        )

        report.append("")


        # Save each preview independently.
        safe_sheet = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            sheet_name
        ).strip("_")


        preview_filename = (
            f"{experiment}_"
            f"{safe_sheet}_"
            f"raw_preview.csv"
        )


        preview.to_csv(
            OUT / preview_filename,
            index=True
        )


        # ====================================================
        # ROW STRUCTURE
        # ====================================================

        for row_idx in range(
            min(40, rows)
        ):

            row = raw.iloc[
                row_idx
            ]


            nonempty = (
                row_nonempty_count(
                    row
                )
            )


            header_like = (
                looks_like_header_row(
                    row
                )
            )


            values = [
                clean_display(v)
                for v in row.tolist()
                if clean_display(v)
            ]


            row_structure_records.append(
                {
                    "experiment":
                        experiment,

                    "workbook":
                        workbook_path.name,

                    "sheet":
                        sheet_name,

                    "row_index":
                        row_idx,

                    "nonempty_cells":
                        nonempty,

                    "header_like":
                        header_like,

                    "values":
                        " | ".join(
                            values[:20]
                        ),
                }
            )


        # ====================================================
        # CELL-LEVEL HEADER / KEYWORD SEARCH
        # ====================================================

        # Inspect first 40 rows because headers may occupy
        # several lines.

        inspect_rows = min(
            40,
            rows
        )


        for r in range(
            inspect_rows
        ):

            for c in range(
                cols
            ):

                value = raw.iat[
                    r,
                    c
                ]


                text = clean_display(
                    value
                )


                if not text:
                    continue


                cell_records.append(
                    {
                        "experiment":
                            experiment,

                        "workbook":
                            workbook_path.name,

                        "sheet":
                            sheet_name,

                        "row_index":
                            r,

                        "column_index":
                            c,

                        "value":
                            text,
                    }
                )


                hits = keyword_hits(
                    text
                )


                for concept in hits:

                    keyword_records.append(
                        {
                            "experiment":
                                experiment,

                            "workbook":
                                workbook_path.name,

                            "sheet":
                                sheet_name,

                            "row_index":
                                r,

                            "column_index":
                                c,

                            "value":
                                text,

                            "concept":
                                concept,
                        }
                    )


        # ====================================================
        # COLUMN VALUE SAMPLES
        # ====================================================

        for c in range(
            cols
        ):

            series = raw.iloc[
                :,
                c
            ]


            nonempty_values = [
                clean_display(v)
                for v in series
                if clean_display(v)
            ]


            sample = (
                nonempty_values[
                    :COLUMN_SAMPLE_SIZE
                ]
            )


            numeric_count = 0


            for value in (
                series.dropna()
            ):

                try:

                    float(value)

                    numeric_count += 1

                except (
                    TypeError,
                    ValueError
                ):
                    pass


            total_nonempty = int(
                series.notna().sum()
            )


            numeric_fraction = (

                numeric_count
                / total_nonempty

                if total_nonempty
                else 0.0
            )


            column_sample_records.append(
                {
                    "experiment":
                        experiment,

                    "workbook":
                        workbook_path.name,

                    "sheet":
                        sheet_name,

                    "column_index":
                        c,

                    "nonempty_count":
                        total_nonempty,

                    "numeric_fraction":
                        numeric_fraction,

                    "sample_values":
                        " | ".join(
                            sample
                        ),
                }
            )


# ============================================================
# CREATE OUTPUT TABLES
# ============================================================

sheet_inventory_df = pd.DataFrame(
    sheet_inventory
)

row_structure_df = pd.DataFrame(
    row_structure_records
)

cells_df = pd.DataFrame(
    cell_records
)

keywords_df = pd.DataFrame(
    keyword_records
)

column_samples_df = pd.DataFrame(
    column_sample_records
)


# ============================================================
# SAVE
# ============================================================

sheet_inventory_df.to_csv(
    OUT
    / "sheet_inventory.csv",
    index=False
)


row_structure_df.to_csv(
    OUT
    / "row_structure.csv",
    index=False
)


cells_df.to_csv(
    OUT
    / "raw_header_cells.csv",
    index=False
)


keywords_df.to_csv(
    OUT
    / "keyword_hits.csv",
    index=False
)


column_samples_df.to_csv(
    OUT
    / "column_samples.csv",
    index=False
)


# ============================================================
# KEYWORD SUMMARY
# ============================================================

report.append("")
report.append("=" * 100)
report.append(
    "KEYWORD / CONCEPT DISCOVERY"
)
report.append("=" * 100)
report.append("")


if not keywords_df.empty:

    keyword_summary = (

        keywords_df

        .groupby(
            [
                "experiment",
                "sheet",
                "concept"
            ]
        )

        .size()

        .reset_index(
            name="hits"
        )

        .sort_values(
            [
                "experiment",
                "sheet",
                "concept"
            ]
        )
    )


    keyword_summary.to_csv(
        OUT
        / "keyword_summary.csv",
        index=False
    )


    report.append(
        keyword_summary.to_string(
            index=False
        )
    )

else:

    keyword_summary = (
        pd.DataFrame()
    )

    report.append(
        "No research keywords detected."
    )


# ============================================================
# POSSIBLE HEADER ROWS
# ============================================================

possible_headers = (

    row_structure_df[
        row_structure_df[
            "header_like"
        ]
    ]

    .copy()
)


possible_headers.to_csv(
    OUT
    / "possible_header_rows.csv",
    index=False
)


report.append("")
report.append("=" * 100)

report.append(
    "POSSIBLE HEADER ROWS "
    "(HEURISTIC ONLY)"
)

report.append("=" * 100)
report.append("")


if not possible_headers.empty:

    report.append(
        possible_headers[
            [
                "experiment",
                "sheet",
                "row_index",
                "nonempty_cells",
                "values"
            ]
        ].to_string(
            index=False
        )
    )

else:

    report.append(
        "No header-like rows detected."
    )


# ============================================================
# RESEARCH STATUS
# ============================================================

report.append("")
report.append("=" * 100)
report.append(
    "RESEARCH DESIGN STATUS"
)
report.append("=" * 100)
report.append("")

report.append(
    "Feature set X_real frozen: NO"
)

report.append(
    "Target Y_real frozen: NO"
)

report.append(
    "Training composition D_real frozen: NO"
)

report.append(
    "Shift family S_real frozen: NO"
)

report.append(
    "Shift severity V_real frozen: NO"
)

report.append(
    "Augmentation A_real frozen: NO"
)

report.append(
    "Model training performed: NO"
)

report.append("")
report.append(
    "This inspection is structural/descriptive only."
)


# ============================================================
# WRITE REPORT
# ============================================================

REPORT_FILE = (
    OUT
    / "phase7_schema_report.txt"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "phase":
        "PHASE7",

    "stage":
        "P7.2_SCHEMA_RECONSTRUCTION",

    "workbooks":
        len(excel_files),

    "sheets":
        len(
            sheet_inventory_df
        ),

    "raw_cells_catalogued":
        len(cells_df),

    "keyword_hits":
        len(keywords_df),

    "possible_header_rows":
        len(possible_headers),

    "preview_rows_per_sheet":
        PREVIEW_ROWS,

    "header_mode":
        None,

    "research_lock": {

        "X_real_frozen":
            False,

        "Y_real_frozen":
            False,

        "D_real_frozen":
            False,

        "S_real_frozen":
            False,

        "V_real_frozen":
            False,

        "A_real_frozen":
            False,

        "model_training":
            False,
    }
}


with open(
    OUT
    / "schema_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# HARD VALIDATION
# ============================================================

expected_experiments = {
    "EXP1",
    "EXP2",
    "EXP3"
}


observed_experiments = set(
    sheet_inventory_df[
        "experiment"
    ].unique()
)


if not (
    expected_experiments
    <= observed_experiments
):

    raise RuntimeError(
        "Schema inspection failed: "
        "not all EXP1/EXP2/EXP3 "
        "workbooks were detected."
    )


if len(
    sheet_inventory_df
) == 0:

    raise RuntimeError(
        "Schema inspection failed: "
        "zero worksheets inspected."
    )


if len(
    cells_df
) == 0:

    raise RuntimeError(
        "Schema inspection failed: "
        "zero raw cells catalogued."
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("PHASE 7.2 SCHEMA INSPECTION SUMMARY")
print("=" * 80)

print(
    f"Workbooks inspected:       "
    f"{len(excel_files)}"
)

print(
    f"Sheets inspected:          "
    f"{len(sheet_inventory_df)}"
)

print(
    f"Raw header cells recorded: "
    f"{len(cells_df)}"
)

print(
    f"Keyword hits:              "
    f"{len(keywords_df)}"
)

print(
    f"Possible header rows:      "
    f"{len(possible_headers)}"
)

print()

print(
    "PHASE 7.2 SCHEMA INSPECTION: PASS"
)

print()

print(
    f"Report:\n{REPORT_FILE}"
)

print()

print(
    "NO MODEL TRAINING WAS PERFORMED."
)

print(
    "NO X/Y/D/S/V/A DEFINITIONS "
    "WERE FROZEN."
)