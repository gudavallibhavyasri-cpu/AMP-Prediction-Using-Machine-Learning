from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DBAASP_CSV = PROJECT_ROOT / "data" / "raw" / "dbaasp.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading DBAASP dataset...")

df = pd.read_csv(DBAASP_CSV)

print("Total records:", len(df))


# --------------------------------------------------
# 3. NORMALISE SEQUENCE AND ANNOTATIONS
# --------------------------------------------------

df["sequence_key"] = (
    df["SEQUENCE"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


def normalise_annotation(value):
    """
    Preserve missing annotations as an explicit category.
    Normalise surrounding whitespace only.
    """
    if pd.isna(value):
        return "[MISSING]"

    value = str(value).strip()

    if not value:
        return "[MISSING]"

    return value.upper()


df["n_terminus_normalised"] = df["N TERMINUS"].apply(
    normalise_annotation
)

df["c_terminus_normalised"] = df["C TERMINUS"].apply(
    normalise_annotation
)


# --------------------------------------------------
# 4. IDENTIFY REPEATED SEQUENCES
# --------------------------------------------------

print("\nIdentifying repeated sequences...")

sequence_counts = (
    df.groupby("sequence_key")
    .size()
)

repeated_sequences = sequence_counts[
    (sequence_counts > 1)
].index

repeated_records = df[
    df["sequence_key"].isin(repeated_sequences)
].copy()


# --------------------------------------------------
# 5. IDENTIFY TERMINAL ANNOTATION CONFLICTS
# --------------------------------------------------

print("\nChecking terminal annotation differences...")

annotation_summary = (
    repeated_records.groupby("sequence_key")
    .agg(
        record_count=("ID", "size"),
        n_terminus_variants=("n_terminus_normalised", "nunique"),
        c_terminus_variants=("c_terminus_normalised", "nunique")
    )
    .reset_index()
)

conflicting_sequences = annotation_summary[
    (
        annotation_summary["n_terminus_variants"] > 1
    )
    |
    (
        annotation_summary["c_terminus_variants"] > 1
    )
].copy()


# --------------------------------------------------
# 6. EXPORT ALL RECORDS FOR CONFLICTING SEQUENCES
# --------------------------------------------------

conflict_details = repeated_records[
    repeated_records["sequence_key"].isin(
        conflicting_sequences["sequence_key"]
    )
].copy()

selected_columns = [
    "ID",
    "NAME",
    "SEQUENCE",
    "sequence_key",
    "N TERMINUS",
    "C TERMINUS",
    "n_terminus_normalised",
    "c_terminus_normalised",
    "SYNTHESIS TYPE",
    "COMPLEXITY",
    "TARGET GROUP",
    "TARGET OBJECT"
]

conflict_details = conflict_details[selected_columns]

conflict_details = conflict_details.sort_values(
    ["sequence_key", "ID"]
)

conflict_details.to_csv(
    RESULTS_DIR / "dbaasp_annotation_conflict_details.csv",
    index=False
)

conflicting_sequences.to_csv(
    RESULTS_DIR / "dbaasp_annotation_conflict_summary.csv",
    index=False
)


# --------------------------------------------------
# 7. CHECK SEQUENCES WITH IDENTICAL ANNOTATIONS
# --------------------------------------------------

same_annotation_sequences = annotation_summary[
    (
        annotation_summary["n_terminus_variants"] == 1
    )
    &
    (
        annotation_summary["c_terminus_variants"] == 1
    )
].copy()

same_annotation_sequences.to_csv(
    RESULTS_DIR / "dbaasp_repeated_sequences_same_annotations.csv",
    index=False
)


# --------------------------------------------------
# 8. PRINT SUMMARY
# --------------------------------------------------

print("\n========== CONFLICT INSPECTION SUMMARY ==========")

print("Unique DBAASP sequence keys:", df["sequence_key"].nunique())

print("Sequences with multiple records:", len(repeated_sequences))

print(
    "Sequences with different terminal annotations:",
    len(conflicting_sequences)
)

print(
    "Sequences with consistent terminal annotations:",
    len(same_annotation_sequences)
)

print("\nExamples of annotation conflicts:")

print(
    conflicting_sequences.head(10).to_string(index=False)
)


# --------------------------------------------------
# 9. COMPLETION
# --------------------------------------------------

print("\n==============================================")
print("DBAASP ANNOTATION CONFLICT INSPECTION COMPLETED")
print("==============================================")

print("\nFiles saved:")
print("- dbaasp_annotation_conflict_details.csv")
print("- dbaasp_annotation_conflict_summary.csv")
print("- dbaasp_repeated_sequences_same_annotations.csv")

print("\nOriginal DBAASP dataset was not modified.")