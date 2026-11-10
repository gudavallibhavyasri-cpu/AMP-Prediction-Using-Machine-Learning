
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_feature_matrix_baseline.csv.gz"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amp_train_test_split.csv"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


print("Loading feature matrix...")

df = pd.read_csv(INPUT_FILE)

print(f"Total sequences: {len(df)}")


# Group sequences by amino-acid composition.
# Sorting the amino acids creates the same key for
# sequences that contain identical amino-acid counts.

print("\nCreating composition-based groups...")

df["composition_group"] = df["sequence"].apply(
    lambda sequence: "".join(sorted(sequence))
)

# Check that every composition group has only one split.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    splitter.split(
        df,
        y=df["label"],
        groups=df["composition_group"]
    )
)

df["split"] = "train"
df.loc[test_indices, "split"] = "test"

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]


# Quality checks
train_groups = set(train_df["composition_group"])
test_groups = set(test_df["composition_group"])

assert train_groups.isdisjoint(test_groups)

assert len(train_df) + len(test_df) == len(df)

assert set(train_df["label"].unique()) == {0, 1}
assert set(test_df["label"].unique()) == {0, 1}


# Save split assignment
df.to_csv(OUTPUT_FILE, index=False)


print("\n========== TRAIN / TEST SPLIT SUMMARY ==========")

print(f"Total sequences: {len(df)}")
print(f"Training sequences: {len(train_df)}")
print(f"Testing sequences: {len(test_df)}")

print(f"\nTraining groups: {len(train_groups)}")
print(f"Testing groups: {len(test_groups)}")

print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test_df["label"].value_counts().sort_index())

print(
    "\nComposition groups appearing in both sets:",
    len(train_groups.intersection(test_groups))
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nLeakage-aware split completed successfully.")
