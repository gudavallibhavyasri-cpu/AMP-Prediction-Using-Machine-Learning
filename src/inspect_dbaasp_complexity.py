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
# 2. LOAD DBAASP
# --------------------------------------------------

print("Loading DBAASP dataset...")

df = pd.read_csv(DBAASP_CSV)

print("Total records:", len(df))


# --------------------------------------------------
# 3. NORMALISE FIELDS
# --------------------------------------------------

def normalise_value(value):
    if pd.isna(value):
        return "[MISSING]"

    value = str(value).strip()

    if not value:
        return "[MISSING]"

    return value


df["complexity_normalised"] = df["COMPLEXITY"].apply(
    normalise_value
)

df["synthesis_normalised"] = df["SYNTHESIS TYPE"].apply(
    normalise_value
)

df["sequence_key"] = (
    df["SEQUENCE"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# --------------------------------------------------
# 4. COMPLEXITY DISTRIBUTION
# --------------------------------------------------

print("\n========== COMPLEXITY DISTRIBUTION ==========")

complexity_counts = (
    df["complexity_normalised"]
    .value_counts(dropna=False)
)

print(complexity_counts.to_string())

complexity_counts.rename_axis("complexity").reset_index(
    name="record_count"
).to_csv(
    RESULTS_DIR / "dbaasp_complexity_distribution.csv",
    index=False
)


# --------------------------------------------------
# 5. SYNTHESIS TYPE DISTRIBUTION
# --------------------------------------------------

print("\n========== SYNTHESIS TYPE DISTRIBUTION ==========")

synthesis_counts = (
    df["synthesis_normalised"]
    .value_counts(dropna=False)
)

print(synthesis_counts.to_string())

synthesis_counts.rename_axis("synthesis_type").reset_index(
    name="record_count"
).to_csv(
    RESULTS_DIR / "dbaasp_synthesis_distribution.csv",
    index=False
)


# --------------------------------------------------
# 6. COMPLEXITY AND SYNTHESIS COMBINATIONS
# --------------------------------------------------

print("\n========== COMPLEXITY × SYNTHESIS TYPE ==========")

combination_table = pd.crosstab(
    df["complexity_normalised"],
    df["synthesis_normalised"],
    margins=True
)

print(combination_table.to_string())

combination_table.to_csv(
    RESULTS_DIR / "dbaasp_complexity_synthesis_combinations.csv"
)


# --------------------------------------------------
# 7. INSPECT EXAMPLE RECORDS
# --------------------------------------------------

example_columns = [
    "ID",
    "NAME",
    "SEQUENCE",
    "COMPLEXITY",
    "SYNTHESIS TYPE",
    "N TERMINUS",
    "C TERMINUS"
]

examples = df[example_columns].copy()

examples.to_csv(
    RESULTS_DIR / "dbaasp_complexity_examples.csv",
    index=False
)


# --------------------------------------------------
# 8. CHECK REPEATED SEQUENCES WITH DIFFERENT
#    COMPLEXITY OR SYNTHESIS ANNOTATIONS
# --------------------------------------------------

print("\n========== REPEATED SEQUENCE ANNOTATIONS ==========")

repeated_mask = df.duplicated(
    subset=["sequence_key"],
    keep=False
)

repeated_records = df[repeated_mask].copy()

annotation_summary = (
    repeated_records.groupby("sequence_key")
    .agg(
        record_count=("ID", "size"),
        complexity_variants=("complexity_normalised", "nunique"),
        synthesis_variants=("synthesis_normalised", "nunique")
    )
    .reset_index()
)

different_annotations = annotation_summary[
    (
        annotation_summary["complexity_variants"] > 1
    )
    |
    (
        annotation_summary["synthesis_variants"] > 1
    )
]

print(
    "Repeated sequences with different complexity or synthesis annotations:",
    len(different_annotations)
)

different_annotations.to_csv(
    RESULTS_DIR / "dbaasp_complexity_annotation_conflicts.csv",
    index=False
)

conflict_details = repeated_records[
    repeated_records["sequence_key"].isin(
        different_annotations["sequence_key"]
    )
][
    [
        "ID",
        "NAME",
        "SEQUENCE",
        "COMPLEXITY",
        "SYNTHESIS TYPE",
        "N TERMINUS",
        "C TERMINUS"
    ]
].sort_values(["SEQUENCE", "ID"])

conflict_details.to_csv(
    RESULTS_DIR / "dbaasp_complexity_conflict_details.csv",
    index=False
)


# --------------------------------------------------
# 9. COMPLETION
# --------------------------------------------------

print("\n==========================================")
print("DBAASP COMPLEXITY INSPECTION COMPLETED")
print("==========================================")

print("\nFiles saved:")
print("- dbaasp_complexity_distribution.csv")
print("- dbaasp_synthesis_distribution.csv")
print("- dbaasp_complexity_synthesis_combinations.csv")
print("- dbaasp_complexity_examples.csv")
print("- dbaasp_complexity_annotation_conflicts.csv")
print("- dbaasp_complexity_conflict_details.csv")

print("\nOriginal DBAASP data was not modified.")