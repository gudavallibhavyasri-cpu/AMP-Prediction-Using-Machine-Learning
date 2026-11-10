
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSITIVE_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_unique_positive_dataset.csv"
)

NEGATIVE_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_synthetic_negative_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_training_dataset_baseline.csv"
)


print("Loading positive and negative datasets...")

positive_df = pd.read_csv(POSITIVE_FILE)
negative_df = pd.read_csv(NEGATIVE_FILE)

# Assign labels and describe the source of each class
positive_df["label"] = 1
positive_df["negative_type"] = "database_amp_record"

negative_df["label"] = 0

# Combine without discarding source metadata
combined_df = pd.concat(
    [positive_df, negative_df],
    ignore_index=True,
    sort=False
)

# Shuffle the records reproducibly
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Quality checks
assert combined_df["sequence"].notna().all()
assert combined_df["sequence"].is_unique
assert set(combined_df["label"].unique()) == {0, 1}

positive_count = (combined_df["label"] == 1).sum()
negative_count = (combined_df["label"] == 0).sum()

# Save dataset
combined_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== COMBINED DATASET SUMMARY ==========")

print(f"Total sequences: {len(combined_df)}")
print(f"Positive sequences: {positive_count}")
print(f"Synthetic negative sequences: {negative_count}")

print("\nClass distribution:")
print(combined_df["label"].value_counts().sort_index())

print("\nNegative sequence types:")
print(
    combined_df.loc[
        combined_df["label"] == 0,
        "negative_type"
    ].value_counts()
)

print("\nDataset columns:")
print(combined_df.columns.tolist())

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nAll quality checks passed.")
