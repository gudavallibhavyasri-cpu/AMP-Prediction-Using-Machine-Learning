
import random
from pathlib import Path

import pandas as pd


# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSITIVE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amp_unique_positive_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amp_synthetic_negative_dataset.csv"
)

# Read positive dataset
print("Loading positive dataset...")

positive_df = pd.read_csv(POSITIVE_FILE)

if "sequence" not in positive_df.columns:
    raise ValueError("The positive dataset must contain a 'sequence' column.")

positive_sequences = (
    positive_df["sequence"]
    .dropna()
    .astype(str)
    .str.upper()
    .unique()
    .tolist()
)

positive_set = set(positive_sequences)

print(f"Unique positive sequences: {len(positive_sequences)}")

# Generate shuffled negative controls
print("\nGenerating synthetic negative sequences...")

negative_sequences = []
negative_set = set()

MAX_ATTEMPTS = 100

for sequence in positive_sequences:

    amino_acids = list(sequence)
    generated = None

    for attempt in range(MAX_ATTEMPTS):

        shuffled = amino_acids.copy()
        random.shuffle(shuffled)

        candidate = "".join(shuffled)

        # Exclude unchanged sequences, known positives,
        # and previously generated negatives.
        if (
            candidate != sequence
            and candidate not in positive_set
            and candidate not in negative_set
        ):
            generated = candidate
            break

    if generated is not None:
        negative_sequences.append(generated)
        negative_set.add(generated)

# Create output dataset
negative_df = pd.DataFrame({
    "sequence": negative_sequences,
    "label": 0,
    "negative_type": "composition_preserving_shuffle"
})

# Quality checks
assert negative_df["sequence"].is_unique
assert not negative_set.intersection(positive_set)

for original, shuffled in zip(
    positive_sequences[:len(negative_sequences)],
    negative_sequences
):
    pass

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
negative_df.to_csv(OUTPUT_FILE, index=False)

# Summary
print("\n========== SYNTHETIC NEGATIVE DATASET ==========")
print(f"Positive sequences available: {len(positive_sequences)}")
print(f"Negative sequences generated: {len(negative_sequences)}")
print(f"Sequences skipped: {len(positive_sequences) - len(negative_sequences)}")
print(f"Duplicate negatives: {negative_df['sequence'].duplicated().sum()}")
print(f"Overlap with positives: {len(negative_set.intersection(positive_set))}")

print("\nNegative sequence examples:")
print(negative_df.head(5).to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nSynthetic negative dataset preparation completed.")
