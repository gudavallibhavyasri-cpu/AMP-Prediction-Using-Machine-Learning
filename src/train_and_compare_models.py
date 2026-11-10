
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amp_train_test_split.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("\nLoading training and testing data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total sequences: {len(df)}")


# ==================================================
# 3. SELECT THE 422 FEATURES
# ==================================================

metadata_columns = [
    "sequence",
    "label",
    "negative_type",
    "split",
    "composition_group"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

if len(feature_columns) != 422:
    raise ValueError(
        f"Expected 422 features, found {len(feature_columns)}."
    )

# Ensure all model features are numeric
X = df[feature_columns].apply(
    pd.to_numeric,
    errors="raise"
)

if X.isnull().any().any():
    raise ValueError("Missing values found in feature columns.")

y = df["label"]

print(f"Number of features: {len(feature_columns)}")


# ==================================================
# 4. CREATE TRAINING AND TESTING SETS
# ==================================================

train_mask = df["split"] == "train"
test_mask = df["split"] == "test"

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ==================================================
# 5. DEFINE MACHINE-LEARNING MODELS
# ==================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}


# ==================================================
# 6. TRAIN AND EVALUATE MODELS
# ==================================================

results = []
prediction_tables = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Store metrics
    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })

    # Display metrics
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    # Save trained model
    model_filename = (
        model_name.lower().replace(" ", "_") + ".pkl"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")

    # Store individual predictions
    prediction_tables[model_name] = pd.DataFrame({
        "sequence": df.loc[test_mask, "sequence"].values,
        "true_label": y_test.values,
        "predicted_label": predictions,
        "predicted_probability": probabilities
    })


# ==================================================
# 7. SAVE MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
).reset_index(drop=True)

comparison_file = RESULTS_DIR / "model_comparison.csv"

results_df.to_csv(
    comparison_file,
    index=False
)


# ==================================================
# 8. SAVE PREDICTIONS
# ==================================================

for model_name, prediction_df in prediction_tables.items():

    filename = (
        model_name.lower().replace(" ", "_")
        + "_predictions.csv"
    )

    prediction_df.to_csv(
        RESULTS_DIR / filename,
        index=False
    )


# ==================================================
# 9. DISPLAY FINAL RESULTS
# ==================================================

print("\n" + "=" * 65)
print("FINAL MACHINE-LEARNING MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

best_model = results_df.iloc[0]["model"]

print("\nBest model based on ROC-AUC:")
print(best_model)

print("\nComparison results saved to:")
print(comparison_file)

print("\nTrained models saved in:")
print(MODEL_DIR)

print("\nIndividual test predictions saved in:")
print(RESULTS_DIR)

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)
