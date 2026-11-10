
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "amp_train_test_split.csv"
)

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading training and testing datasets...")

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

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="raise"
)

y = df["label"]

groups = df["composition_group"]

train_mask = df["split"] == "train"
test_mask = df["split"] == "test"

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

groups_train = groups.loc[train_mask]

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")
print(f"Features: {len(feature_columns)}")


# ==================================================
# 3. GROUPED CROSS-VALIDATION
# ==================================================

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==================================================
# 4. RANDOM FOREST HYPERPARAMETER SEARCH
# ==================================================

print("\nPreparing Random Forest hyperparameter search...")

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

parameter_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=parameter_grid,
    n_iter=15,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)


# ==================================================
# 5. RUN SEARCH USING TRAINING DATA ONLY
# ==================================================

print("\nStarting hyperparameter optimisation...")
print("The search may take several minutes.")

search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\nHyperparameter optimisation completed.")


# ==================================================
# 6. DISPLAY BEST PARAMETERS
# ==================================================

print("\n" + "=" * 60)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 60)

print(search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(f"{search.best_score_:.4f}")


# ==================================================
# 7. EVALUATE BEST MODEL ON HELD-OUT TEST SET
# ==================================================

best_model = search.best_estimator_

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:, 1]

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


# ==================================================
# 8. PRINT TEST RESULTS
# ==================================================

print("\n" + "=" * 60)
print("OPTIMISED RANDOM FOREST - TEST RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
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


# ==================================================
# 9. SAVE OPTIMISED MODEL
# ==================================================

MODEL_FILE = MODEL_DIR / "random_forest_optimized.pkl"

joblib.dump(best_model, MODEL_FILE)

print("\nOptimised model saved to:")
print(MODEL_FILE)


# ==================================================
# 10. SAVE SEARCH RESULTS
# ==================================================

search_results = pd.DataFrame(search.cv_results_)

search_results = search_results.sort_values(
    by="rank_test_score"
)

SEARCH_RESULTS_FILE = (
    RESULTS_DIR / "random_forest_hyperparameter_search.csv"
)

search_results.to_csv(
    SEARCH_RESULTS_FILE,
    index=False
)

test_results = pd.DataFrame([{
    "model": "Optimised Random Forest",
    "best_cv_roc_auc": search.best_score_,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc
}])

TEST_RESULTS_FILE = (
    RESULTS_DIR / "random_forest_optimized_metrics.csv"
)

test_results.to_csv(
    TEST_RESULTS_FILE,
    index=False
)


# ==================================================
# 11. SAVE TEST PREDICTIONS
# ==================================================

prediction_results = pd.DataFrame({
    "sequence": df.loc[test_mask, "sequence"].values,
    "true_label": y_test.values,
    "predicted_label": predictions,
    "predicted_probability": probabilities
})

prediction_results.to_csv(
    RESULTS_DIR / "random_forest_optimized_predictions.csv",
    index=False
)


print("\nSearch results saved to:")
print(SEARCH_RESULTS_FILE)

print("\nTest metrics saved to:")
print(TEST_RESULTS_FILE)

print("\nTest predictions saved successfully.")

print("\n" + "=" * 60)
print("EXPERIMENT 3 COMPLETED SUCCESSFULLY")
print("=" * 60)
