
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from feature_extraction import (
    extract_features,
    FEATURE_NAMES,
    AA_LIST
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "cleaned_apd3.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "amp_model_baseline.pkl"


# ==========================================
# 2. RANDOM SEED
# ==========================================

RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


# ==========================================
# 3. LOAD POSITIVE AMP SEQUENCES
# ==========================================

print("\n1. LOADING APD3 DATASET")

df = pd.read_csv(DATA_FILE)

positive_sequences = (
    df["sequence"]
    .astype(str)
    .str.strip()
    .str.upper()
    .tolist()
)

if not positive_sequences:
    raise ValueError("No positive sequences found.")

if len(set(positive_sequences)) != len(positive_sequences):
    raise ValueError("Duplicate positive sequences detected.")

print("Positive AMP sequences:", len(positive_sequences))


# ==========================================
# 4. GENERATE LENGTH-MATCHED SYNTHETIC
#    NEGATIVE SEQUENCES
# ==========================================

print("\n2. GENERATING SYNTHETIC NEGATIVES")

positive_set = set(positive_sequences)
used_negatives = set()

negative_sequences = []

for positive_seq in positive_sequences:

    seq_length = len(positive_seq)

    while True:

        negative_seq = "".join(
            rng.choice(AA_LIST, size=seq_length)
        )

        if (
            negative_seq not in positive_set
            and negative_seq not in used_negatives
        ):
            used_negatives.add(negative_seq)
            negative_sequences.append(negative_seq)
            break

print("Negative sequences:", len(negative_sequences))

assert len(negative_sequences) == len(positive_sequences)


# ==========================================
# 5. COMBINE DATA AND CREATE LABELS
# ==========================================

print("\n3. COMBINING DATA")

all_sequences = positive_sequences + negative_sequences

labels = np.array(
    [1] * len(positive_sequences)
    + [0] * len(negative_sequences)
)

print("Total sequences:", len(all_sequences))
print("Positive samples:", sum(labels == 1))
print("Negative samples:", sum(labels == 0))


# ==========================================
# 6. FEATURE EXTRACTION
# ==========================================

print("\n4. EXTRACTING 422 FEATURES")

X = np.array(
    [
        extract_features(seq)
        for seq in all_sequences
    ],
    dtype=np.float64
)

y = labels

assert X.shape[1] == 422

print("Feature matrix shape:", X.shape)


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

print("\n5. SPLITTING DATASET")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. RANDOM FOREST CLASSIFIER
# ==========================================

print("\n6. TRAINING RANDOM FOREST")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ==========================================
# 9. PROBABILITY CALIBRATION
# ==========================================

print("\n7. CALIBRATING MODEL")

model = CalibratedClassifierCV(
    estimator=rf,
    method="sigmoid",
    cv=3
)

model.fit(X_train, y_train)


# ==========================================
# 10. MODEL PREDICTION
# ==========================================

print("\n8. EVALUATING MODEL")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ==========================================
# 11. PERFORMANCE METRICS
# ==========================================

auc = roc_auc_score(y_test, probabilities)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print("ROC-AUC:", round(auc, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Synthetic negative", "APD3 positive"],
    zero_division=0
))


# ==========================================
# 12. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, MODEL_FILE)

print("\nModel saved:", MODEL_FILE)


# ==========================================
# 13. SAVE PERFORMANCE METRICS
# ==========================================

metrics = pd.DataFrame([{
    "model": "Calibrated Random Forest",
    "positive_samples": len(positive_sequences),
    "negative_samples": len(negative_sequences),
    "features": X.shape[1],
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "roc_auc": auc,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

metrics.to_csv(
    RESULTS_DIR / "baseline_metrics.csv",
    index=False
)


# ==========================================
# 14. SAVE TEST PREDICTIONS
# ==========================================

_, test_sequences, _, _ = train_test_split(
    all_sequences,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

prediction_results = pd.DataFrame({
    "sequence": test_sequences,
    "true_label": y_test,
    "predicted_probability": probabilities,
    "predicted_label": predictions
})

prediction_results.to_csv(
    RESULTS_DIR / "baseline_predictions.csv",
    index=False
)


print("\n" + "=" * 50)
print("TRAINING PIPELINE COMPLETED")
print("=" * 50)
