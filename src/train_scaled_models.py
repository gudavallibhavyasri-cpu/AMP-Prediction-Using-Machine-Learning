
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "amp_train_test_split.csv"
)

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

metadata_columns = [
    "sequence",
    "label",
    "negative_type",
    "split",
    "composition_group"
]

feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

if len(feature_columns) != 422:
    raise ValueError(
        f"Expected 422 features, found {len(feature_columns)}"
    )

X = df[feature_columns].apply(pd.to_numeric, errors="raise")
y = df["label"]

train_mask = df["split"] == "train"
test_mask = df["split"] == "test"

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

test_sequences = df.loc[test_mask, "sequence"].reset_index(drop=True)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features: {len(feature_columns)}")


# ==========================================
# 3. DEFINE MODELS
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression Scaled": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "SVM Scaled": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=False,
            random_state=42
        ))
    ])
}


# ==========================================
# 4. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Use probability scores when available.
    # For SVM, use decision_function for ROC-AUC.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

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
    auc = roc_auc_score(y_test, scores)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save model separately; preserve Experiment 1 files.
    filename = (
        name.lower().replace(" ", "_")
        + "_experiment2.pkl"
    )

    joblib.dump(model, MODEL_DIR / filename)

    # Save predictions
    prediction_df = pd.DataFrame({
        "sequence": test_sequences,
        "true_label": y_test.reset_index(drop=True),
        "predicted_label": predictions,
        "prediction_score": scores
    })

    prediction_filename = (
        name.lower().replace(" ", "_")
        + "_experiment2_predictions.csv"
    )

    prediction_df.to_csv(
        RESULTS_DIR / prediction_filename,
        index=False
    )


# ==========================================
# 5. SAVE COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

OUTPUT_FILE = RESULTS_DIR / "model_comparison_experiment2.csv"

results_df.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 65)
print("EXPERIMENT 2 - MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nBest model by ROC-AUC:")
print(results_df.iloc[0]["model"])

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nExperiment 2 completed successfully.")
