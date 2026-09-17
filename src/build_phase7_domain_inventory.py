"""
Phase 7.3 — Real Domain Inventory (FINAL FIXED VERSION)
=======================================================

Purpose
-------
Reconstruct the real experimental-domain structure from HydroGrowNet
harvest worksheets across EXP1, EXP2, and EXP3.

Handles both layouts:
- EXP1/EXP2: domain label -> header -> plant rows
- EXP3:      header -> domain label + first plant row -> remaining plant rows

This script DOES NOT:
- train models
- augment data
- select/freeze predictors or targets
- define/freeze shift families or severity
- modify source files
- automatically delete duplicates

Expected source:
data/phase7/raw/all_months_sensory_data.zip

Outputs:
results/experiment_7/domain_inventory/
"""

from pathlib import Path
import zipfile
import shutil
import re
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "phase7" / "raw"
ZIP_FILE = RAW_DIR / "all_months_sensory_data.zip"

OUT = ROOT / "results" / "experiment_7" / "domain_inventory"
EXTRACT_DIR = OUT / "raw_excel"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = value.replace("\r", " ")
    value = re.sub(r"\s+", " ", value)
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


def is_harvest_sheet(sheet_name):
    name = sheet_name.lower()
    return "harvest" in name and "measurement" in name


# ============================================================
# DOMAIN LABEL PARSER
# ============================================================

def parse_domain_label(value):
    """
    Parse labels such as:
        R1-T1
        R1 T1
        R1T1
        Replicate 1 T1
        Replicate 1 Treatment 1
    """

    text = clean_text(value)
    if not text:
        return None

    normalized = text.upper().replace("_", "-")
    normalized = re.sub(r"\s+", " ", normalized)

    patterns = [
        r"\bR\s*(\d+)\s*[- ]?\s*T\s*(\d+)\b",
        r"\bREPLICATE\s*(\d+)\s*[- ]?\s*T\s*(\d+)\b",
        r"\bREPLICATE\s*(\d+).*?TREATMENT\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)

        if match:
            replicate = int(match.group(1))
            treatment = int(match.group(2))

            return {
                "replicate": replicate,
                "treatment": treatment,
                "normalized_domain": f"R{replicate}-T{treatment}",
            }

    return None


# ============================================================
# HEADER DETECTION
# ============================================================

def looks_like_harvest_header(row):
    values = [clean_text(v).lower() for v in row.tolist()]
    joined = " | ".join(values)

    plant_hit = (
        "plant no" in joined
        or "plant no." in joined
        or "plant number" in joined
    )

    outcome_hits = sum(
        keyword in joined
        for keyword in [
            "total weight",
            "plant height",
            "shoot length",
            "root length",
            "head diameter",
            "h.d",
            "hd (cm)",
            "stem diameter",
            "shoot weight",
            "root weight",
            "no. of leaves",
            "no.of leaves",
            "number of leaves",
        ]
    )

    return plant_hit and outcome_hits >= 2


def find_header_near_domain(
    raw,
    domain_row,
    backward_search=3,
    forward_search=5,
):
    """
    EXP1/EXP2:
        domain label
        header
        plant rows

    EXP3:
        header
        domain label + first plant row
        remaining plant rows

    Search both before and after the domain-label row.
    """

    # EXP3-style: header before domain row.
    start = max(0, domain_row - backward_search)

    for row_idx in range(domain_row - 1, start - 1, -1):
        if looks_like_harvest_header(raw.iloc[row_idx]):
            return row_idx

    # EXP1/EXP2-style: header after domain row.
    end = min(len(raw), domain_row + forward_search + 1)

    for row_idx in range(domain_row + 1, end):
        if looks_like_harvest_header(raw.iloc[row_idx]):
            return row_idx

    return None


# ============================================================
# STANDARDIZE HARVEST COLUMN NAMES
# ============================================================

def standardize_column_name(value):
    name = clean_text(value).lower()
    name = name.replace(".", "")
    name = re.sub(r"\s+", " ", name)

    mappings = [
        (["plant no", "plant number"], "plant_no"),
        (["total weight"], "total_weight_g"),
        (["plant height"], "plant_height_cm"),
        (["shoot length"], "shoot_length_cm"),
        (["root length"], "root_length_cm"),

        # EXP1/EXP2: "Head diameter"
        # EXP3: "H.D (cm)" -> after period removal becomes "hd (cm)"
        (["head diameter", "hd"], "head_diameter_raw"),

        (["stem diameter"], "stem_diameter_cm"),

        (
            ["shoot weight before removing", "shoot weight before"],
            "shoot_weight_before_g",
        ),

        (
            ["shoot weight after removing", "shoot weight after"],
            "shoot_weight_after_g",
        ),

        (["shoot weight"], "shoot_weight_g"),
        (["root weight"], "root_weight_g"),

        (
            ["no of leaves", "number of leaves"],
            "n_leaves",
        ),
    ]

    for patterns, standard in mappings:
        if any(pattern in name for pattern in patterns):
            return standard

    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")

    return name if name else "unnamed"


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def numeric_or_nan(value):
    if pd.isna(value):
        return np.nan

    text = clean_text(value)

    if not text:
        return np.nan

    if text.upper() in {"-", "--", "NA", "N/A", "NAN"}:
        return np.nan

    try:
        return float(text)
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# HEAD-DIAMETER PARSER
# ============================================================

def parse_head_diameter(value):
    """
    Preserve values such as 24*27 and derive the two axes and their mean.
    The '*' is a dimension separator, NOT multiplication.
    """

    text = clean_text(value)

    if not text:
        return np.nan, np.nan, np.nan

    text = (
        text.replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
    )

    match = re.match(
        r"^\s*([0-9]*\.?[0-9]+)\s*\*\s*([0-9]*\.?[0-9]+)\s*$",
        text,
    )

    if match:
        a = float(match.group(1))
        b = float(match.group(2))
        return a, b, (a + b) / 2.0

    try:
        number = float(text)
        return number, number, number
    except ValueError:
        return np.nan, np.nan, np.nan


# ============================================================
# FIND HARVEST DOMAIN BLOCKS
# ============================================================

def find_domain_blocks(raw):
    blocks = []

    for row_idx in range(len(raw)):
        row = raw.iloc[row_idx]

        for col_idx, value in enumerate(row.tolist()):
            parsed = parse_domain_label(value)

            if parsed is None:
                continue

            blocks.append(
                {
                    "row_index": row_idx,
                    "column_index": col_idx,
                    "raw_label": clean_text(value),
                    **parsed,
                }
            )

    unique = {}

    for block in blocks:
        key = (block["row_index"], block["normalized_domain"])
        unique[key] = block

    blocks = list(unique.values())
    blocks.sort(key=lambda x: (x["row_index"], x["column_index"]))

    return blocks


# ============================================================
# EXTRACT ONE HARVEST BLOCK
# ============================================================

def extract_block(
    raw,
    block,
    next_block_row,
    experiment,
    workbook_name,
    sheet_name,
):
    header_row = find_header_near_domain(
        raw,
        block["row_index"],
    )

    if header_row is None:
        return [], {
            "status": "HEADER_NOT_FOUND",
            "experiment": experiment,
            "sheet": sheet_name,
            "domain": block["normalized_domain"],
            "raw_label": block["raw_label"],
            "domain_row": block["row_index"],
            "header_row": np.nan,
            "end_row": np.nan,
            "plants_extracted": np.nan,
        }

    header_values = raw.iloc[header_row].tolist()

    column_names = [
        standardize_column_name(value)
        for value in header_values
    ]

    # Make duplicate standardized names unique.
    seen = {}
    unique_columns = []

    for name in column_names:
        if name not in seen:
            seen[name] = 0
            unique_columns.append(name)
        else:
            seen[name] += 1
            unique_columns.append(f"{name}_{seen[name]}")

    end_row = next_block_row if next_block_row is not None else len(raw)

    # EXP3: header precedes domain row, and domain row contains plant 1.
    # EXP1/EXP2: header follows domain row, and plant data starts after header.
    if header_row < block["row_index"]:
        data_start_row = block["row_index"]
    else:
        data_start_row = header_row + 1

    records = []

    for row_idx in range(data_start_row, end_row):
        values = raw.iloc[row_idx].tolist()
        row_dict = dict(zip(unique_columns, values))

        plant_raw = row_dict.get("plant_no", np.nan)
        plant_no = numeric_or_nan(plant_raw)

        # Blank/summary rows do not have a valid plant number.
        if pd.isna(plant_no):
            continue

        if not float(plant_no).is_integer():
            continue

        plant_no = int(plant_no)

        record = {
            "experiment": experiment,
            "workbook": workbook_name,
            "sheet": sheet_name,
            "domain": block["normalized_domain"],
            "replicate": block["replicate"],
            "treatment": block["treatment"],
            "plant_no": plant_no,
            "source_row": row_idx,
        }

        numeric_fields = [
            "total_weight_g",
            "plant_height_cm",
            "shoot_length_cm",
            "root_length_cm",
            "stem_diameter_cm",
            "shoot_weight_before_g",
            "shoot_weight_after_g",
            "shoot_weight_g",
            "root_weight_g",
            "n_leaves",
        ]

        for field in numeric_fields:
            if field in row_dict:
                record[field] = numeric_or_nan(row_dict[field])

        if "head_diameter_raw" in row_dict:
            raw_diameter = row_dict["head_diameter_raw"]
            a, b, mean = parse_head_diameter(raw_diameter)

            record["head_diameter_raw"] = clean_text(raw_diameter)
            record["head_diameter_axis1_cm"] = a
            record["head_diameter_axis2_cm"] = b
            record["head_diameter_mean_cm"] = mean

        records.append(record)

    diagnostic = {
        "status": "OK",
        "experiment": experiment,
        "sheet": sheet_name,
        "domain": block["normalized_domain"],
        "raw_label": block["raw_label"],
        "domain_row": block["row_index"],
        "header_row": header_row,
        "end_row": end_row,
        "plants_extracted": len(records),
    }

    return records, diagnostic


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PHASE 7.3 — REAL DOMAIN INVENTORY")
print("=" * 80)

if not ZIP_FILE.exists():
    candidates = sorted(RAW_DIR.glob("*.zip"))

    if len(candidates) == 1:
        ZIP_FILE = candidates[0]
    else:
        raise FileNotFoundError(
            "Could not locate Phase 7 ZIP archive."
        )

print(f"\nArchive: {ZIP_FILE.name}")


# ============================================================
# EXTRACT ARCHIVE COPY
# ============================================================

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    z.extractall(EXTRACT_DIR)

excel_files = sorted(EXTRACT_DIR.rglob("*.xlsx"))

if not excel_files:
    raise RuntimeError("No Excel files found.")


# ============================================================
# PROCESS WORKBOOKS
# ============================================================

all_records = []
block_diagnostics = []
sheet_records = []

for workbook_path in excel_files:
    experiment = detect_experiment(workbook_path.name)

    print()
    print("-" * 80)
    print(experiment, workbook_path.name)
    print("-" * 80)

    workbook = pd.ExcelFile(
        workbook_path,
        engine="openpyxl",
    )

    harvest_sheets = [
        sheet
        for sheet in workbook.sheet_names
        if is_harvest_sheet(sheet)
    ]

    print("Harvest sheets:", harvest_sheets)

    for sheet_name in harvest_sheets:
        raw = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
        )

        blocks = find_domain_blocks(raw)

        print()
        print(f"Sheet: {sheet_name}")
        print(f"Raw shape: {raw.shape}")
        print(
            "Domain labels:",
            [block["normalized_domain"] for block in blocks],
        )

        sheet_records.append(
            {
                "experiment": experiment,
                "workbook": workbook_path.name,
                "sheet": sheet_name,
                "raw_rows": len(raw),
                "raw_columns": len(raw.columns),
                "domain_labels_found": len(blocks),
            }
        )

        for i, block in enumerate(blocks):
            next_row = (
                blocks[i + 1]["row_index"]
                if i + 1 < len(blocks)
                else None
            )

            records, diagnostic = extract_block(
                raw=raw,
                block=block,
                next_block_row=next_row,
                experiment=experiment,
                workbook_name=workbook_path.name,
                sheet_name=sheet_name,
            )

            all_records.extend(records)
            block_diagnostics.append(diagnostic)


# ============================================================
# CREATE TABLES
# ============================================================

plants_df = pd.DataFrame(all_records)
blocks_df = pd.DataFrame(block_diagnostics)
sheets_df = pd.DataFrame(sheet_records)


# ============================================================
# HARD VALIDATION
# ============================================================

if plants_df.empty:
    raise RuntimeError(
        "No harvest plant records were reconstructed."
    )

required = {
    "experiment",
    "domain",
    "replicate",
    "treatment",
    "plant_no",
}

missing_required = required - set(plants_df.columns)

if missing_required:
    raise RuntimeError(
        f"Missing required columns: {missing_required}"
    )


# ============================================================
# UNIQUE PLANT KEY
# ============================================================

plants_df["plant_key"] = (
    plants_df["experiment"].astype(str)
    + "_"
    + plants_df["domain"].astype(str)
    + "_P"
    + plants_df["plant_no"].astype(str)
)

duplicate_plant_keys = int(
    plants_df["plant_key"].duplicated().sum()
)


# ============================================================
# DOMAIN COUNTS
# ============================================================

domain_counts = (
    plants_df
    .groupby(
        [
            "experiment",
            "treatment",
            "replicate",
            "domain",
        ],
        dropna=False,
    )
    .agg(
        n_rows=("plant_key", "size"),
        n_unique_plants=("plant_key", "nunique"),
    )
    .reset_index()
)


# ============================================================
# EXPERIMENT COUNTS
# ============================================================

experiment_counts = (
    plants_df
    .groupby("experiment")
    .agg(
        n_rows=("plant_key", "size"),
        n_unique_plants=("plant_key", "nunique"),
        n_domains=("domain", "nunique"),
        n_treatments=("treatment", "nunique"),
        n_replicates=("replicate", "nunique"),
    )
    .reset_index()
)


# ============================================================
# OUTCOME AVAILABILITY
# ============================================================

metadata_columns = {
    "experiment",
    "workbook",
    "sheet",
    "domain",
    "replicate",
    "treatment",
    "plant_no",
    "source_row",
    "plant_key",
    "head_diameter_raw",
}

outcome_columns = [
    col
    for col in plants_df.columns
    if col not in metadata_columns
]

outcome_records = []

for experiment in sorted(plants_df["experiment"].unique()):
    subset = plants_df[
        plants_df["experiment"] == experiment
    ]

    for column in outcome_columns:
        if column not in subset.columns:
            continue

        nonmissing = int(subset[column].notna().sum())

        outcome_records.append(
            {
                "experiment": experiment,
                "outcome": column,
                "n_total": len(subset),
                "n_nonmissing": nonmissing,
                "fraction_available": (
                    nonmissing / len(subset)
                    if len(subset)
                    else np.nan
                ),
            }
        )

outcome_availability = pd.DataFrame(outcome_records)


# ============================================================
# TOTAL WEIGHT SUMMARY
# ============================================================

weight_records = []

if "total_weight_g" in plants_df.columns:
    for keys, subset in plants_df.groupby(
        ["experiment", "domain"]
    ):
        experiment, domain = keys

        values = pd.to_numeric(
            subset["total_weight_g"],
            errors="coerce",
        )

        valid = values.dropna()

        weight_records.append(
            {
                "experiment": experiment,
                "domain": domain,
                "n": len(valid),
                "mean_total_weight_g": (
                    valid.mean() if len(valid) else np.nan
                ),
                "std_total_weight_g": (
                    valid.std() if len(valid) > 1 else np.nan
                ),
                "min_total_weight_g": (
                    valid.min() if len(valid) else np.nan
                ),
                "max_total_weight_g": (
                    valid.max() if len(valid) else np.nan
                ),
            }
        )

weight_summary = pd.DataFrame(weight_records)


# ============================================================
# CROSS-EXPERIMENT DOMAIN COVERAGE
# ============================================================

coverage = (
    domain_counts
    .pivot_table(
        index=[
            "treatment",
            "replicate",
            "domain",
        ],
        columns="experiment",
        values="n_unique_plants",
        fill_value=0,
    )
    .reset_index()
)

coverage.columns.name = None


# ============================================================
# SAVE OUTPUTS
# ============================================================

plants_df.to_csv(
    OUT / "harvest_plant_inventory.csv",
    index=False,
)

blocks_df.to_csv(
    OUT / "harvest_block_diagnostics.csv",
    index=False,
)

sheets_df.to_csv(
    OUT / "harvest_sheet_inventory.csv",
    index=False,
)

domain_counts.to_csv(
    OUT / "domain_counts.csv",
    index=False,
)

experiment_counts.to_csv(
    OUT / "experiment_counts.csv",
    index=False,
)

outcome_availability.to_csv(
    OUT / "outcome_availability.csv",
    index=False,
)

weight_summary.to_csv(
    OUT / "total_weight_by_domain.csv",
    index=False,
)

coverage.to_csv(
    OUT / "cross_experiment_domain_coverage.csv",
    index=False,
)


# ============================================================
# REPORT
# ============================================================

report = []

report.append("PHASE 7.3 — REAL DOMAIN INVENTORY")
report.append("=" * 90)
report.append("")

report.append(
    f"Harvest plant rows reconstructed: {len(plants_df)}"
)
report.append(
    f"Unique plant keys: {plants_df['plant_key'].nunique()}"
)
report.append(
    f"Duplicate plant keys: {duplicate_plant_keys}"
)
report.append("")

report.append("EXPERIMENT COUNTS")
report.append("-" * 90)
report.append(experiment_counts.to_string(index=False))
report.append("")

report.append("DOMAIN COUNTS")
report.append("-" * 90)
report.append(domain_counts.to_string(index=False))
report.append("")

report.append("CROSS-EXPERIMENT DOMAIN COVERAGE")
report.append("-" * 90)
report.append(coverage.to_string(index=False))
report.append("")

report.append("OUTCOME AVAILABILITY")
report.append("-" * 90)
report.append(outcome_availability.to_string(index=False))
report.append("")

if not weight_summary.empty:
    report.append("TOTAL WEIGHT BY DOMAIN")
    report.append("-" * 90)
    report.append(weight_summary.to_string(index=False))
    report.append("")

report.append("BLOCK DIAGNOSTICS")
report.append("-" * 90)
report.append(blocks_df.to_string(index=False))
report.append("")

report.append("RESEARCH LOCK STATUS")
report.append("-" * 90)
report.append("Primary target frozen: NO")
report.append("Feature set frozen: NO")
report.append("Domain definition frozen: NO")
report.append("Shift family frozen: NO")
report.append("Shift severity frozen: NO")
report.append("Augmentation frozen: NO")
report.append("Model training performed: NO")

REPORT_FILE = OUT / "phase7_domain_inventory_report.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "phase": "PHASE7",
    "stage": "P7.3_DOMAIN_INVENTORY_FIXED",
    "harvest_rows": int(len(plants_df)),
    "unique_plants": int(plants_df["plant_key"].nunique()),
    "duplicate_plant_keys": duplicate_plant_keys,
    "domains": int(
        domain_counts[
            ["experiment", "domain"]
        ]
        .drop_duplicates()
        .shape[0]
    ),
    "experiments": sorted(
        plants_df["experiment"].unique().tolist()
    ),
    "model_training": False,
    "research_variables_frozen": False,
}

with open(
    OUT / "domain_inventory_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)


# ============================================================
# FINAL VALIDATION
# ============================================================

observed_experiments = set(
    plants_df["experiment"].unique()
)

expected_experiments = {
    "EXP1",
    "EXP2",
    "EXP3",
}

all_blocks_ok = (
    not blocks_df.empty
    and blocks_df["status"].eq("OK").all()
)

all_expected_experiments = (
    expected_experiments <= observed_experiments
)

if not all_expected_experiments:
    print()
    print("WARNING:")
    print(
        "Harvest records were not reconstructed "
        "for all three experiments."
    )
    print(
        "Observed:",
        sorted(observed_experiments),
    )

if not all_blocks_ok:
    print()
    print("WARNING:")
    print(
        "At least one harvest domain block "
        "did not parse successfully."
    )

if duplicate_plant_keys > 0:
    print()
    print("WARNING:")
    print(
        f"{duplicate_plant_keys} duplicate plant keys detected."
    )
    print("Do NOT remove them automatically.")


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("PHASE 7.3 DOMAIN INVENTORY SUMMARY")
print("=" * 80)

print(
    f"Harvest rows reconstructed: {len(plants_df)}"
)
print(
    f"Unique plant keys:          "
    f"{plants_df['plant_key'].nunique()}"
)
print(
    f"Experiment-domain cells:    "
    f"{len(domain_counts)}"
)
print(
    f"Duplicate plant keys:       "
    f"{duplicate_plant_keys}"
)
print(
    f"Experiments recovered:      "
    f"{', '.join(sorted(observed_experiments))}"
)
print(
    f"All domain blocks OK:       "
    f"{all_blocks_ok}"
)

print()

if (
    all_expected_experiments
    and all_blocks_ok
    and duplicate_plant_keys == 0
):
    print("PHASE 7.3 DOMAIN INVENTORY: PASS")
else:
    print("PHASE 7.3 DOMAIN INVENTORY: REVIEW REQUIRED")

print()
print(f"Report:\n{REPORT_FILE}")
print()
print("NO MODEL TRAINING WAS PERFORMED.")
print(
    "NO X/Y/D/S/V/A DEFINITIONS WERE FROZEN."
)
