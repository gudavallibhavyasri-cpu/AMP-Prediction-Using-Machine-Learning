
import pandas as pd
import numpy as np
import joblib
import random

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from feature_extraction import extract_features


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = ROOT / "models" / "amp_model_baseline.pkl"
TRAIN_FILE = ROOT / "data" / "cleaned_apd3.csv"
VALIDATION_FILE = ROOT / "validation" / "validation_amps.txt"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD MODEL AND TRAINING DATA
# ==========================================

print("\n1. LOADING MODEL")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully")

training_df = pd.read_csv(TRAIN_FILE)

training_sequences = set(
    training_df["sequence"].astype(str).str.strip().str.upper()
)


# ==========================================
# 3. LOAD VALIDATION AMPS
# ==========================================

print("\n2. LOADING VALIDATION SEQUENCES")

with open(VALIDATION_FILE, "r") as file:
    amps = [
        line.strip().upper()
        for line in file
        if line.strip()
    ]

print("Total validation sequences:", len(amps))

if not amps:
    raise ValueError("No validation sequences found.")

if len(set(amps)) != len(amps):
    raise ValueError("Duplicate validation sequences found.")

# Validate all sequences
for seq in amps:
    extract_features(seq)


# ==========================================
# 4. CHECK TRAINING DATA OVERLAP
# ==========================================

print("\n3. CHECKING TRAINING OVERLAP")

overlapping = [
    seq for seq in amps
    if seq in training_sequences
]

independent_amps = [
    seq for seq in amps
    if seq not in training_sequences
]

print("Overlapping sequences:", len(overlapping))
print("Non-overlapping sequences:", len(independent_amps))

if overlapping:
    print("\nOverlapping AMP sequences:")
    for seq in overlapping:
        print(seq)

if not independent_amps:
    raise ValueError(
        "No non-overlapping AMP sequences available."
    )


# ==========================================
# 5. GENERATE SCRAMBLED CONTROLS
# ==========================================

print("\n4. GENERATING SCRAMBLED CONTROLS")

rng = random.Random(42)

negatives = []

for seq in independent_amps:

    shuffled = list(seq)

    for attempt in range(1000):

        rng.shuffle(shuffled)
        candidate = "".join(shuffled)

        if (
            candidate != seq
            and candidate not in training_sequences
            and candidate not in negatives
        ):
            negatives.append(candidate)
            break

    else:
        raise RuntimeError(
            f"Could not generate a unique control for {seq}"
        )

print("Positive controls:", len(independent_amps))
print("Scrambled controls:", len(negatives))


# ==========================================
# 6. COMBINE VALIDATION DATA
# ==========================================

all_sequences = independent_amps + negatives

true_labels = np.array(
    [1] * len(independent_amps)
    + [0] * len(negatives)
)

print("\n5. EXTRACTING FEATURES")

X = np.array(
    [extract_features(seq) for seq in all_sequences],
    dtype=float
)

print("Feature matrix shape:", X.shape)


# ==========================================
# 7. PREDICT USING TRAINED MODEL
# ==========================================

print("\n6. MODEL PREDICTION")

probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ==========================================
# 8. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(true_labels, predictions)

precision = precision_score(
    true_labels, predictions, zero_division=0
)

recall = recall_score(
    true_labels, predictions, zero_division=0
)

f1 = f1_score(
    true_labels, predictions, zero_division=0
)

auc = roc_auc_score(true_labels, probabilities)

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

print("\n" + "=" * 50)
print("INDEPENDENT CONTROL VALIDATION")
print("=" * 50)

print("ROC-AUC:", round(auc, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 9. SAVE DETAILED PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "sequence": all_sequences,
    "control_type": (
        ["Known AMP"] * len(independent_amps)
        + ["Shuffled control"] * len(negatives)
    ),
    "true_label": true_labels,
    "AMP_probability": probabilities,
    "predicted_label": predictions
})

results.to_csv(
    RESULTS_DIR / "validation_results.csv",
    index=False
)


# ==========================================
# 10. SAVE METRICS
# ==========================================

metrics = pd.DataFrame([{
    "validation_AMPs": len(independent_amps),
    "shuffled_controls": len(negatives),
    "training_overlaps_excluded": len(overlapping),
    "ROC_AUC": auc,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_score": f1
}])

metrics.to_csv(
    RESULTS_DIR / "validation_metrics.csv",
    index=False
)


print("\nValidation results saved.")
print("Results folder:", RESULTS_DIR)
print("\nVALIDATION COMPLETED")
