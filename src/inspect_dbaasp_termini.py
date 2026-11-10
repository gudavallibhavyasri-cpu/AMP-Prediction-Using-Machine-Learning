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

print(f"Total records: {len(df)}")


# --------------------------------------------------
# 3. INSPECT TERMINAL FIELDS
# --------------------------------------------------

terminal_columns = [
    "N TERMINUS",
    "C TERMINUS"
]

for column in terminal_columns:

    print(f"\n========== {column} ==========")

    non_missing = df[column].dropna().astype(str).str.strip()

    print("Non-missing records:", len(non_missing))
    print("Distinct values:", non_missing.nunique())

    print("\nMost frequent values:")
    print(non_missing.value_counts().head(20).to_string())


# --------------------------------------------------
# 4. CHECK MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUE SUMMARY ==========")

for column in terminal_columns:
    missing = df[column].isna().sum()
    print(f"{column}: {missing} missing values")


# --------------------------------------------------
# 5. INSPECT EXAMPLE RECORDS
# --------------------------------------------------

print("\n========== EXAMPLE RECORDS ==========")

example_columns = [
    "ID",
    "NAME",
    "SEQUENCE",
    "N TERMINUS",
    "C TERMINUS",
    "SYNTHESIS TYPE"
]

examples = df[example_columns].copy()

examples.to_csv(
    RESULTS_DIR / "dbaasp_terminal_examples.csv",
    index=False
)

print(examples.head(30).to_string(index=False))


# --------------------------------------------------
# 6. CHECK SEQUENCES WITH MULTIPLE TERMINAL ANNOTATIONS
# --------------------------------------------------

print("\n========== TERMINAL ANNOTATION CONSISTENCY ==========")

df["sequence_key"] = (
    df["SEQUENCE"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

annotation_counts = (
    df.groupby("sequence_key")
    .agg(
        record_count=("ID", "size"),
        n_terminus_variants=("N TERMINUS", "nunique"),
        c_terminus_variants=("C TERMINUS", "nunique")
    )
    .reset_index()
)

multiple_annotations = annotation_counts[
    (annotation_counts["record_count"] > 1)
    & (
        (annotation_counts["n_terminus_variants"] > 1)
        | (annotation_counts["c_terminus_variants"] > 1)
    )
]

print(
    "Sequences with multiple records and different terminal annotations:",
    len(multiple_annotations)
)

multiple_annotations.to_csv(
    RESULTS_DIR / "dbaasp_multiple_terminal_annotations.csv",
    index=False
)


# --------------------------------------------------
# 7. SAVE COMPLETE TERMINAL FIELD SUMMARY
# --------------------------------------------------

terminal_summary = []

for column in terminal_columns:

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    counts = values.value_counts()

    for value, count in counts.items():

        terminal_summary.append({
            "field": column,
            "annotation": value,
            "record_count": count
        })

terminal_summary_df = pd.DataFrame(terminal_summary)

terminal_summary_df.to_csv(
    RESULTS_DIR / "dbaasp_terminal_annotation_summary.csv",
    index=False
)


# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\n==========================================")
print("DBAASP TERMINAL INSPECTION COMPLETED")
print("==========================================")

print("\nFiles saved:")
print("- dbaasp_terminal_examples.csv")
print("- dbaasp_multiple_terminal_annotations.csv")
print("- dbaasp_terminal_annotation_summary.csv")

print("\nOriginal DBAASP data was not modified.")