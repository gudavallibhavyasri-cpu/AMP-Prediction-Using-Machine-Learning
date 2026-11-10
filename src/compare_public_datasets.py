from pathlib import Path
import pandas as pd
import csv
import re


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APD3_CSV = PROJECT_ROOT / "data" / "cleaned_apd3.csv"
DBAASP_CSV = PROJECT_ROOT / "data" / "raw" / "dbaasp.csv"
DRAMP_EXCEL = PROJECT_ROOT / "data" / "raw" / "dramp_general.xlsx"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. SEQUENCE VALIDATION
# --------------------------------------------------

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence):
    """
    Remove surrounding whitespace and convert to uppercase.
    Do not replace ambiguous or modified amino acid symbols.
    """
    if pd.isna(sequence):
        return ""

    return str(sequence).strip().upper()


def sequence_status(sequence):
    if not sequence:
        return "empty"

    if not re.fullmatch(r"[A-Z]+", sequence):
        return "contains_non_letter_characters"

    if not set(sequence).issubset(STANDARD_AA):
        return "contains_nonstandard_amino_acids"

    if len(sequence) < 5:
        return "shorter_than_5"

    return "standard"


# --------------------------------------------------
# 3. LOAD APD3
# --------------------------------------------------

print("\nLoading APD3...")

apd3 = pd.read_csv(APD3_CSV)

apd3_data = pd.DataFrame({
    "database": "APD3",
    "record_id": apd3["id"].astype(str),
    "sequence": apd3["sequence"].apply(clean_sequence)
})


# --------------------------------------------------
# 4. LOAD DBAASP
# --------------------------------------------------

print("\nLoading DBAASP...")

dbaasp = pd.read_csv(DBAASP_CSV)

dbaasp_data = pd.DataFrame({
    "database": "DBAASP",
    "record_id": dbaasp["ID"].astype(str),
    "sequence": dbaasp["SEQUENCE"].apply(clean_sequence)
})


# --------------------------------------------------
# 5. LOAD DRAMP
# --------------------------------------------------

print("\nLoading DRAMP...")

dramp = pd.read_excel(DRAMP_EXCEL, sheet_name="general_amps")

dramp_data = pd.DataFrame({
    "database": "DRAMP",
    "record_id": dramp["DRAMP_ID"].astype(str),
    "sequence": dramp["Sequence"].apply(clean_sequence)
})


# --------------------------------------------------
# 6. COMBINE DATASETS
# --------------------------------------------------

combined = pd.concat(
    [apd3_data, dbaasp_data, dramp_data],
    ignore_index=True
)

combined["length"] = combined["sequence"].str.len()

combined["sequence_status"] = combined["sequence"].apply(
    sequence_status
)


# --------------------------------------------------
# 7. DATASET SUMMARY
# --------------------------------------------------

summary = (
    combined.groupby(["database", "sequence_status"])
    .size()
    .unstack(fill_value=0)
)

summary["total_records"] = summary.sum(axis=1)

print("\n========== DATASET QUALITY SUMMARY ==========")
print(summary)

summary.to_csv(RESULTS_DIR / "public_dataset_quality_summary.csv")


# --------------------------------------------------
# 8. EXACT SEQUENCE OVERLAP
# --------------------------------------------------

print("\nChecking exact sequence overlap...")

valid_sequences = combined[
    combined["sequence_status"] == "standard"
].copy()

database_sets = {
    database: set(group["sequence"])
    for database, group in valid_sequences.groupby("database")
}

overlap_rows = []

database_names = ["APD3", "DBAASP", "DRAMP"]

for i in range(len(database_names)):
    for j in range(i + 1, len(database_names)):

        db1 = database_names[i]
        db2 = database_names[j]

        shared = database_sets[db1].intersection(database_sets[db2])

        overlap_rows.append({
            "database_1": db1,
            "database_2": db2,
            "shared_unique_sequences": len(shared)
        })

        print(f"{db1} vs {db2}: {len(shared)} shared sequences")


overlap_df = pd.DataFrame(overlap_rows)

overlap_df.to_csv(
    RESULTS_DIR / "exact_sequence_overlap.csv",
    index=False
)


# --------------------------------------------------
# 9. DUPLICATE SEQUENCES WITHIN EACH DATABASE
# --------------------------------------------------

print("\nChecking duplicates within databases...")

duplicate_summary = (
    valid_sequences.groupby("database")
    .agg(
        total_valid_records=("sequence", "size"),
        unique_sequences=("sequence", "nunique")
    )
)

duplicate_summary["duplicate_records"] = (
    duplicate_summary["total_valid_records"]
    - duplicate_summary["unique_sequences"]
)

print(duplicate_summary)

duplicate_summary.to_csv(
    RESULTS_DIR / "within_database_duplicates.csv"
)


# --------------------------------------------------
# 10. SAVE COMBINED SEQUENCE INVENTORY
# --------------------------------------------------

combined.to_csv(
    RESULTS_DIR / "combined_dataset_inventory.csv",
    index=False
)


# --------------------------------------------------
# 11. SAVE NON-STANDARD SEQUENCES FOR REVIEW
# --------------------------------------------------

nonstandard = combined[
    combined["sequence_status"] != "standard"
].copy()

nonstandard.to_csv(
    RESULTS_DIR / "sequences_requiring_review.csv",
    index=False
)


# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\n============================================")
print("PUBLIC DATASET COMPARISON COMPLETED")
print("============================================")

print("\nFiles saved in the results folder:")
print("- public_dataset_quality_summary.csv")
print("- exact_sequence_overlap.csv")
print("- within_database_duplicates.csv")
print("- combined_dataset_inventory.csv")
print("- sequences_requiring_review.csv")

print("\nIMPORTANT:")
print("Original raw datasets were not modified.")
print("Non-standard sequences were retained for review.")
print("No sequences were silently deleted or converted.")