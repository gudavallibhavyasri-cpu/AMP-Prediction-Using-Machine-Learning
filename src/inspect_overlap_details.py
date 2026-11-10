from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APD3_CSV = PROJECT_ROOT / "data" / "cleaned_apd3.csv"
DBAASP_CSV = PROJECT_ROOT / "data" / "raw" / "dbaasp.csv"
DRAMP_EXCEL = PROJECT_ROOT / "data" / "raw" / "dramp_general.xlsx"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("Loading APD3...")
apd3 = pd.read_csv(APD3_CSV)

print("Loading DBAASP...")
dbaasp = pd.read_csv(DBAASP_CSV)

print("Loading DRAMP...")
dramp = pd.read_excel(
    DRAMP_EXCEL,
    sheet_name="general_amps"
)


# --------------------------------------------------
# 3. STANDARDISE SEQUENCE REPRESENTATION
# --------------------------------------------------

def normalise_sequence(sequence):
    if pd.isna(sequence):
        return ""

    return str(sequence).strip().upper()


apd3["sequence_key"] = apd3["sequence"].apply(normalise_sequence)
dbaasp["sequence_key"] = dbaasp["SEQUENCE"].apply(normalise_sequence)
dramp["sequence_key"] = dramp["Sequence"].apply(normalise_sequence)


def is_standard_sequence(sequence):
    return (
        len(sequence) >= 5
        and bool(sequence)
        and set(sequence).issubset(STANDARD_AA)
    )


# --------------------------------------------------
# 4. CREATE RECORD TABLES
# --------------------------------------------------

apd3_records = pd.DataFrame({
    "database": "APD3",
    "record_id": apd3["id"].astype(str),
    "sequence": apd3["sequence_key"]
})

dbaasp_records = pd.DataFrame({
    "database": "DBAASP",
    "record_id": dbaasp["ID"].astype(str),
    "sequence": dbaasp["sequence_key"],
    "name": dbaasp["NAME"],
    "complexity": dbaasp["COMPLEXITY"],
    "n_terminus": dbaasp["N TERMINUS"],
    "c_terminus": dbaasp["C TERMINUS"],
    "synthesis_type": dbaasp["SYNTHESIS TYPE"],
    "target_group": dbaasp["TARGET GROUP"],
    "target_object": dbaasp["TARGET OBJECT"]
})

dramp_records = pd.DataFrame({
    "database": "DRAMP",
    "record_id": dramp["DRAMP_ID"].astype(str),
    "sequence": dramp["sequence_key"],
    "name": dramp["Name"],
    "source": dramp["Source"],
    "activity": dramp["Activity"],
    "target_organism": dramp["Target_Organism"],
    "hemolytic_activity": dramp["Hemolytic_activity"],
    "cytotoxicity": dramp["Cytotoxicity"],
    "modification": dramp["Other_Modifications"],
    "pubmed_id": dramp["Pubmed_ID"],
    "reference": dramp["Reference"]
})


# --------------------------------------------------
# 5. COMBINE RECORDS
# --------------------------------------------------

all_records = pd.concat(
    [apd3_records, dbaasp_records, dramp_records],
    ignore_index=True,
    sort=False
)

all_records["is_standard"] = all_records["sequence"].apply(
    is_standard_sequence
)

standard_records = all_records[
    all_records["is_standard"]
].copy()


# --------------------------------------------------
# 6. IDENTIFY CROSS-DATABASE OVERLAPS
# --------------------------------------------------

database_count = (
    standard_records.groupby("sequence")["database"]
    .nunique()
)

overlapping_sequences = database_count[
    database_count > 1
].index

overlap_details = standard_records[
    standard_records["sequence"].isin(overlapping_sequences)
].copy()

overlap_details["database_count"] = (
    overlap_details["sequence"].map(database_count)
)

overlap_details = overlap_details.sort_values(
    ["sequence", "database"]
)

overlap_details.to_csv(
    RESULTS_DIR / "cross_database_overlap_details.csv",
    index=False
)


# --------------------------------------------------
# 7. IDENTIFY WITHIN-DATABASE DUPLICATES
# --------------------------------------------------

duplicate_mask = standard_records.duplicated(
    subset=["database", "sequence"],
    keep=False
)

within_database_duplicates = standard_records[
    duplicate_mask
].sort_values(["database", "sequence"])

within_database_duplicates.to_csv(
    RESULTS_DIR / "within_database_duplicate_details.csv",
    index=False
)


# # --------------------------------------------------
# 8. CREATE A UNIQUE SEQUENCE SUMMARY
# --------------------------------------------------

def combine_unique_values(values):
    """
    Convert values to strings, ignore missing values,
    remove duplicates and sort them safely.
    """
    unique_values = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }

    return "; ".join(sorted(unique_values))


unique_sequence_summary = (
    standard_records.groupby("sequence")
    .agg(
        number_of_records=("record_id", "size"),
        number_of_databases=("database", "nunique"),
        databases=("database", combine_unique_values),
        record_ids=("record_id", combine_unique_values)
    )
    .reset_index()
)

unique_sequence_summary["sequence_length"] = (
    unique_sequence_summary["sequence"].str.len()
)

unique_sequence_summary = unique_sequence_summary[
    [
        "sequence",
        "sequence_length",
        "number_of_records",
        "number_of_databases",
        "databases",
        "record_ids"
    ]
]

unique_sequence_summary.to_csv(
    RESULTS_DIR / "unique_sequence_source_summary.csv",
    index=False
)

# --------------------------------------------------
# 9. PRINT SUMMARY
# --------------------------------------------------

print("\n========== OVERLAP DETAILS ==========")

print(f"Total standard records: {len(standard_records)}")

print(
    "Unique standard sequences:",
    standard_records["sequence"].nunique()
)

print(
    "Sequences present in multiple databases:",
    len(overlapping_sequences)
)

print(
    "Records involved in cross-database overlaps:",
    len(overlap_details)
)

print(
    "Records involved in within-database duplicates:",
    len(within_database_duplicates)
)

print("\nOverlapping sequences by database combination:")

for sequence in overlapping_sequences:
    databases = sorted(
        standard_records.loc[
            standard_records["sequence"] == sequence,
            "database"
        ].unique()
    )

    combination = " + ".join(databases)

    print(f"{combination}: {sequence}")


# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n========================================")
print("OVERLAP INSPECTION COMPLETED")
print("========================================")

print("\nFiles saved:")
print("- cross_database_overlap_details.csv")
print("- within_database_duplicate_details.csv")
print("- unique_sequence_source_summary.csv")

print("\nOriginal datasets were not modified.")