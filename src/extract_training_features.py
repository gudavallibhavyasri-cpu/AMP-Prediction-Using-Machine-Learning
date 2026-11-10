
from pathlib import Path

import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_training_dataset_baseline.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_feature_matrix_baseline.csv.gz"
)


print("Loading combined training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Sequences loaded: {len(df)}")

# Extract the 422 features for each sequence
print("\nExtracting sequence features...")

feature_rows = []

for sequence in df["sequence"]:
    features = extract_features(sequence)

    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, "
            f"but received {len(features)}."
        )

    feature_rows.append(features)

features_df = pd.DataFrame(
    feature_rows,
    columns=FEATURE_NAMES
)

# Retain identifiers and labels separately from model features
output_df = pd.concat(
    [
        df[["sequence", "label", "negative_type"]].reset_index(drop=True),
        features_df.reset_index(drop=True)
    ],
    axis=1
)

# Quality checks
if output_df.isnull().any().any():
    raise ValueError("Missing values found in the feature matrix.")

if output_df.shape[1] != 425:
    raise ValueError(
        f"Expected 425 columns, found {output_df.shape[1]}."
    )

if len(output_df) != len(df):
    raise ValueError("Feature extraction changed the number of records.")

# Save compressed feature matrix
output_df.to_csv(OUTPUT_FILE, index=False, compression="gzip")

print("\n========== FEATURE EXTRACTION SUMMARY ==========")
print(f"Total sequences: {len(output_df)}")
print(f"Number of features: {len(FEATURE_NAMES)}")
print(f"Total columns including sequence and labels: {output_df.shape[1]}")
print(f"Missing values: {output_df.isnull().sum().sum()}")

print("\nClass distribution:")
print(output_df["label"].value_counts().sort_index())

print("\nFirst five feature names:")
print(FEATURE_NAMES[:5])

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFeature extraction completed successfully.")
