
from pathlib import Path
import pandas as pd


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amp_master_dataset.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 5


# ==================================================
# 2. LOAD MASTER DATASET
# ==================================================

print("\nLoading master dataset...")

if not MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset not found: {MASTER_FILE}"
    )

df = pd.read_csv(MASTER_FILE)

print("Total source records:", len(df))


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def normalise_sequence(sequence):

    if pd.isna(sequence):
        return ""

    return str(sequence).strip().upper()


def normalise_annotation(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    missing_values = {
        "",
        "nan",
        "none",
        "not included yet",
        "not available",
        "not found",
        "not included",
        "unknown"
    }

    if value.lower() in missing_values:
        return ""

    return value


def combine_annotations(values):

    """
    Safely combine unique non-missing annotations.
    """

    unique_values = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }

    return "; ".join(sorted(unique_values))


def is_standard_sequence(sequence):

    if not sequence:
        return False

    return (
        len(sequence) >= MIN_LENGTH
        and set(sequence).issubset(STANDARD_AA)
    )


# ==================================================
# 4. IDENTIFY CHEMICAL MODIFICATIONS
# ==================================================

def has_reported_modification(row):

    modification_columns = [
        "n_terminal_modification",
        "c_terminal_modification",
        "other_modifications"
    ]

    for column in modification_columns:

        value = normalise_annotation(row.get(column))

        if value:

            if value.upper() not in {
                "FREE",
                "NO SPECIFIC N-TERMINAL",
                "NO SPECIFIC C-TERMINAL"
            }:
                return True

    # Check stereochemistry
    stereochemistry = normalise_annotation(
        row.get("stereochemistry")
    )

    if stereochemistry:
        if stereochemistry.upper() != "L":
            return True

    # Check DBAASP complexity
    complexity = normalise_annotation(
        row.get("complexity")
    )

    if complexity:
        if complexity.upper() != "MONOMER":
            return True

    # Check DRAMP peptide structure
    structure = normalise_annotation(
        row.get("structure")
    )

    if structure:
        if structure.lower() not in {"linear"}:
            return True

    return False


def classify_chemistry(group):

    """
    Classify chemistry using available database annotations.

    Missing metadata is not proof that a peptide is unmodified.
    """

    if any(
        has_reported_modification(row)
        for _, row in group.iterrows()
    ):
        return "modification_reported"

    databases = set(group["database"].dropna())

    if databases == {"APD3"}:
        return "not_reported_in_APD3_metadata"

    # Check whether modification fields are sufficiently annotated.
    chemistry_fields = [
        "n_terminal_modification",
        "c_terminal_modification",
        "other_modifications",
        "stereochemistry",
        "complexity",
        "structure"
    ]

    has_any_chemistry_metadata = False

    for column in chemistry_fields:

        if column not in group.columns:
            continue

        if any(
            normalise_annotation(value)
            for value in group[column]
        ):
            has_any_chemistry_metadata = True
            break

    if has_any_chemistry_metadata:
        return "no_modification_reported_in_available_metadata"

    return "incomplete_chemistry_metadata"


# ==================================================
# 5. STANDARDISE SEQUENCES
# ==================================================

print("\nChecking sequence quality...")

df["sequence"] = df["sequence"].apply(
    normalise_sequence
)

df["sequence_is_standard"] = df["sequence"].apply(
    is_standard_sequence
)

valid_records = df[
    (df["initial_eligibility"] == "eligible")
    & (df["sequence_is_standard"])
].copy()

print("Eligible source records:", len(valid_records))

print(
    "Unique standard sequences:",
    valid_records["sequence"].nunique()
)


# ==================================================
# 6. GROUP RECORDS BY EXACT SEQUENCE
# ==================================================

print("\nGrouping identical sequences...")

grouped = valid_records.groupby(
    "sequence",
    sort=True
)


# ==================================================
# 7. BUILD UNIQUE SEQUENCE DATASET
# ==================================================

unique_rows = []

for sequence, group in grouped:

    unique_rows.append({

        "sequence": sequence,

        "sequence_length": len(sequence),

        "number_of_source_records": len(group),

        "number_of_source_databases": (
            group["database"].nunique()
        ),

        "source_databases": combine_annotations(
            group["database"]
        ),

        "source_record_ids": combine_annotations(
    group["database"].astype(str)
    + ":"
    + group["record_id"].fillna("UNKNOWN").astype(str)
),

        "chemistry_status": classify_chemistry(group),

        "source_names": combine_annotations(
            group["name"]
        ),

        "activity_annotations": combine_annotations(
            group["activity"]
        ),

        "target_organisms": combine_annotations(
            group["target_organism"]
        ),

        "references": combine_annotations(
            group["reference"]
        )
    })


unique_dataset = pd.DataFrame(unique_rows)


# ==================================================
# 8. ADD POSITIVE CLASS LABEL
# ==================================================

unique_dataset["label"] = 1

unique_dataset["label_description"] = (
    "AMP database record; positive class"
)

unique_dataset["model_feature_type"] = (
    "standard amino acid sequence features"
)


# ==================================================
# 9. SAVE UNIQUE POSITIVE DATASET
# ==================================================

print("\nSaving unique positive dataset...")

unique_file = (
    PROCESSED_DIR / "amp_unique_positive_dataset.csv"
)

unique_dataset.to_csv(
    unique_file,
    index=False
)


# ==================================================
# 10. SAVE CHEMISTRY REVIEW DATASETS
# ==================================================

modified = unique_dataset[
    unique_dataset["chemistry_status"]
    == "modification_reported"
].copy()

incomplete = unique_dataset[
    unique_dataset["chemistry_status"].isin([
        "not_reported_in_APD3_metadata",
        "incomplete_chemistry_metadata"
    ])
].copy()

no_modification_reported = unique_dataset[
    unique_dataset["chemistry_status"]
    == "no_modification_reported_in_available_metadata"
].copy()


modified.to_csv(
    RESULTS_DIR / "amp_sequences_with_reported_modifications.csv",
    index=False
)

incomplete.to_csv(
    RESULTS_DIR / "amp_sequences_with_incomplete_chemistry_metadata.csv",
    index=False
)

no_modification_reported.to_csv(
    RESULTS_DIR / "amp_sequences_without_reported_modifications.csv",
    index=False
)


# ==================================================
# 11. SOURCE DATABASE SUMMARY
# ==================================================

source_summary = (
    unique_dataset.groupby("source_databases")
    .size()
    .reset_index(name="unique_sequence_count")
)

source_summary.to_csv(
    RESULTS_DIR / "unique_sequences_by_source_combination.csv",
    index=False
)


# ==================================================
# 12. CHEMISTRY SUMMARY
# ==================================================

chemistry_summary = (
    unique_dataset["chemistry_status"]
    .value_counts()
    .rename_axis("chemistry_status")
    .reset_index(name="sequence_count")
)

chemistry_summary.to_csv(
    RESULTS_DIR / "chemistry_status_summary.csv",
    index=False
)


# ==================================================
# 13. FINAL DATASET QUALITY CHECKS
# ==================================================

assert unique_dataset["sequence"].is_unique, (
    "Duplicate sequences remain in the unique dataset."
)

assert unique_dataset["sequence"].apply(
    is_standard_sequence
).all(), "Invalid sequences found in the final dataset."

assert unique_dataset["label"].eq(1).all(), (
    "Unexpected labels found."
)


# ==================================================
# 14. PRINT FINAL SUMMARY
# ==================================================

print("\n========== FINAL POSITIVE DATASET SUMMARY ==========")

print("Unique positive sequences:", len(unique_dataset))

print(
    "Unique sequences from multiple databases:",
    (unique_dataset["number_of_source_databases"] > 1).sum()
)

print("\nChemistry status distribution:")

print(
    unique_dataset["chemistry_status"]
    .value_counts()
    .to_string()
)

print("\nSource database combinations:")

print(source_summary.to_string(index=False))


# ==================================================
# 15. COMPLETION
# ==================================================

print("\n==========================================")
print("TRAINING DATASET PREPARATION COMPLETED")
print("==========================================")

print("\nGenerated files:")

print("data/processed/amp_unique_positive_dataset.csv")

print("results/amp_sequences_with_reported_modifications.csv")

print("results/amp_sequences_with_incomplete_chemistry_metadata.csv")

print("results/amp_sequences_without_reported_modifications.csv")

print("results/unique_sequences_by_source_combination.csv")

print("results/chemistry_status_summary.csv")

print("\nQuality checks passed.")
print("Original raw datasets were not modified.")
print("No negative sequences were generated.")
