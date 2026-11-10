
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    ROOT / "models"
    / "random_forest_optimized.pkl"
)

DATA_FILE = (
    ROOT / "data" / "processed"
    / "amp_train_test_split.csv"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD MODEL AND DATA
# ==================================================

print("\nLoading optimised Random Forest model...")

model = joblib.load(MODEL_FILE)

df = pd.read_csv(DATA_FILE)

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

# Use the held-out test set
test_df = df[df["split"] == "test"].copy()

X_test = test_df[feature_columns].apply(
    pd.to_numeric,
    errors="raise"
)

y_test = test_df["label"]

print(f"Test sequences: {len(X_test)}")
print(f"Features: {len(feature_columns)}")


# ==================================================
# 3. BASELINE ROC-AUC
# ==================================================

print("\nCalculating baseline ROC-AUC...")

baseline_probabilities = model.predict_proba(X_test)[:, 1]

baseline_auc = roc_auc_score(
    y_test,
    baseline_probabilities
)

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")


# ==================================================
# 4. PERMUTATION IMPORTANCE
# ==================================================

print("\nStarting permutation importance analysis...")
print("This may take several minutes.")

permutation_results = permutation_importance(
    estimator=model,
    X=X_test,
    y=y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

print("\nPermutation analysis completed.")


# ==================================================
# 5. CREATE IMPORTANCE TABLE
# ==================================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": permutation_results.importances_mean,
    "importance_std": permutation_results.importances_std
})

importance_df = importance_df.sort_values(
    by="importance_mean",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = range(
    1,
    len(importance_df) + 1
)

importance_file = (
    RESULTS_DIR
    / "permutation_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ==================================================
# 6. DISPLAY TOP 20 FEATURES
# ==================================================

top_20 = importance_df.head(20)

print("\n" + "=" * 65)
print("TOP 20 FEATURES - PERMUTATION IMPORTANCE")
print("=" * 65)

print(
    top_20[
        ["rank", "feature", "importance_mean", "importance_std"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ==================================================
# 7. PLOT TOP 20 FEATURES
# ==================================================

plt.figure(figsize=(12, 8))

plt.barh(
    top_20["feature"][::-1],
    top_20["importance_mean"][::-1],
    xerr=top_20["importance_std"][::-1]
)

plt.xlabel("Mean decrease in ROC-AUC")
plt.ylabel("Features")

plt.title(
    "Top 20 Permutation Feature Importances"
)

plt.tight_layout()

plot_file = (
    RESULTS_DIR
    / "permutation_feature_importance_top20.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 8. FEATURE CATEGORY ANALYSIS
# ==================================================

def get_feature_category(feature):

    if feature.startswith("AAC_"):
        return "Amino Acid Composition"

    elif feature.startswith("DPC_"):
        return "Dipeptide Composition"

    elif feature in [
        "Average_Hydrophobicity",
        "hydrophobicity"
    ]:
        return "Hydrophobicity"

    elif feature in [
        "Approximate_Net_Charge",
        "net_charge"
    ]:
        return "Net Charge"

    else:
        return "Other"


importance_df["category"] = (
    importance_df["feature"].apply(
        get_feature_category
    )
)

category_importance = (
    importance_df.groupby("category")["importance_mean"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

category_importance.columns = [
    "feature_category",
    "summed_permutation_importance"
]

category_file = (
    RESULTS_DIR
    / "permutation_feature_category_importance.csv"
)

category_importance.to_csv(
    category_file,
    index=False
)


# ==================================================
# 9. DISPLAY CATEGORY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("PERMUTATION IMPORTANCE BY CATEGORY")
print("=" * 65)

print(
    category_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 65)

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")

print("\nGenerated files:")
print(importance_file)
print(plot_file)
print(category_file)

print("\nAll analysis steps completed successfully.")
