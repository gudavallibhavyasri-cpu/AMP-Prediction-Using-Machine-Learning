from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DRAMP_EXCEL = PROJECT_ROOT / "data" / "raw" / "dramp_general.xlsx"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD DRAMP
# --------------------------------------------------

print("Loading DRAMP dataset...")

df = pd.read_excel(
    DRAMP_EXCEL,
    sheet_name="general_amps"
)

print("Total records:", len(df))
print("Total columns:", len(df.columns))


# --------------------------------------------------
# 3. ACTIVITY DISTRIBUTION
# --------------------------------------------------

print("\n========== ACTIVITY DISTRIBUTION ==========")

activity_counts = (
    df["Activity"]
    .fillna("[MISSING]")
    .astype(str)
    .str.strip()
    .value_counts()
)

print(activity_counts.head(25).to_string())

activity_counts.rename_axis("activity").reset_index(
    name="record_count"
).to_csv(
    RESULTS_DIR / "dramp_activity_distribution.csv",
    index=False
)


# --------------------------------------------------
# 4. TARGET ORGANISM DISTRIBUTION
# --------------------------------------------------

print("\n========== TARGET ORGANISM DISTRIBUTION ==========")

target_counts = (
    df["Target_Organism"]
    .fillna("[MISSING]")
    .astype(str)
    .str.strip()
    .value_counts()
)

print(target_counts.head(20).to_string())

target_counts.rename_axis("target_organism").reset_index(
    name="record_count"
).to_csv(
    RESULTS_DIR / "dramp_target_distribution.csv",
    index=False
)


# --------------------------------------------------
# 5. CYTOTOXICITY AND HEMOLYTIC ANNOTATIONS
# --------------------------------------------------

annotation_columns = [
    "Cytotoxicity",
    "Hemolytic_activity",
    "Protein_existence",
    "Linear/Cyclic/Branched"
]

annotation_summary = []

print("\n========== BIOLOGICAL ANNOTATIONS ==========")

for column in annotation_columns:

    counts = (
        df[column]
        .fillna("[MISSING]")
        .astype(str)
        .str.strip()
        .value_counts()
    )

    print(f"\n{column}:")
    print(counts.head(15).to_string())

    for value, count in counts.items():
        annotation_summary.append({
            "field": column,
            "annotation": value,
            "record_count": count
        })


pd.DataFrame(annotation_summary).to_csv(
    RESULTS_DIR / "dramp_biological_annotation_summary.csv",
    index=False
)


# --------------------------------------------------
# 6. SEQUENCE LENGTH CONSISTENCY
# --------------------------------------------------

print("\n========== SEQUENCE LENGTH CHECK ==========")

df["calculated_length"] = (
    df["Sequence"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.len()
)

df["length_matches"] = (
    df["calculated_length"]
    == pd.to_numeric(df["Sequence_Length"], errors="coerce")
)

print("Matching lengths:", df["length_matches"].sum())
print("Mismatched lengths:", (~df["length_matches"]).sum())

length_mismatches = df[
    ~df["length_matches"]
][
    ["DRAMP_ID", "Sequence", "Sequence_Length", "calculated_length"]
]

length_mismatches.to_csv(
    RESULTS_DIR / "dramp_length_mismatches.csv",
    index=False
)


# --------------------------------------------------
# 7. MODIFICATION ANNOTATIONS
# --------------------------------------------------

print("\n========== MODIFICATION ANNOTATIONS ==========")

modification_columns = [
    "N-terminal_Modification",
    "C-terminal_Modification",
    "Other_Modifications",
    "Stereochemistry"
]

modification_summary = []

for column in modification_columns:

    values = (
        df[column]
        .fillna("[MISSING]")
        .astype(str)
        .str.strip()
        .value_counts()
    )

    print(f"\n{column}:")
    print(values.head(10).to_string())

    for value, count in values.items():
        modification_summary.append({
            "field": column,
            "annotation": value,
            "record_count": count
        })


pd.DataFrame(modification_summary).to_csv(
    RESULTS_DIR / "dramp_modification_summary.csv",
    index=False
)


# --------------------------------------------------
# 8. EXPORT IMPORTANT METADATA
# --------------------------------------------------

metadata_columns = [
    "DRAMP_ID",
    "Sequence",
    "Sequence_Length",
    "Name",
    "Source",
    "Activity",
    "Target_Organism",
    "Cytotoxicity",
    "Hemolytic_activity",
    "Linear/Cyclic/Branched",
    "N-terminal_Modification",
    "C-terminal_Modification",
    "Other_Modifications",
    "Stereochemistry",
    "Pubmed_ID",
    "Reference"
]

metadata = df[metadata_columns].copy()

metadata.to_csv(
    RESULTS_DIR / "dramp_metadata_inventory.csv",
    index=False
)


# --------------------------------------------------
# 9. COMPLETION
# --------------------------------------------------

print("\n==========================================")
print("DRAMP INSPECTION COMPLETED")
print("==========================================")

print("\nFiles saved:")

print("- dramp_activity_distribution.csv")
print("- dramp_target_distribution.csv")
print("- dramp_biological_annotation_summary.csv")
print("- dramp_length_mismatches.csv")
print("- dramp_modification_summary.csv")
print("- dramp_metadata_inventory.csv")

print("\nOriginal DRAMP data was not modified.")