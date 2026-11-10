
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


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

print(f"Features loaded: {len(feature_columns)}")


# ==================================================
# 3. EXTRACT FEATURE IMPORTANCE
# ==================================================

print("\nCalculating feature importance...")

importance_values = model.feature_importances_

if len(importance_values) != len(feature_columns):
    raise ValueError(
        "Feature importance count does not match feature count."
    )

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = range(
    1,
    len(importance_df) + 1
)


# ==================================================
# 4. SAVE ALL FEATURE IMPORTANCE VALUES
# ==================================================

importance_file = (
    RESULTS_DIR
    / "random_forest_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print("\nFeature importance saved to:")
print(importance_file)


# ==================================================
# 5. DISPLAY TOP 20 FEATURES
# ==================================================

top_20 = importance_df.head(20)

print("\n" + "=" * 60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("=" * 60)

print(
    top_20[["rank", "feature", "importance"]].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ==================================================
# 6. PLOT TOP 20 FEATURES
# ==================================================

plt.figure(figsize=(12, 8))

plt.barh(
    top_20["feature"][::-1],
    top_20["importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")

plt.title(
    "Top 20 Feature Importances - Optimised Random Forest"
)

plt.tight_layout()

plot_file = (
    RESULTS_DIR
    / "random_forest_top20_feature_importance.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTop 20 feature plot saved to:")
print(plot_file)


# ==================================================
# 7. CLASSIFY FEATURES INTO CATEGORIES
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


# ==================================================
# 8. CALCULATE CATEGORY-WISE IMPORTANCE
# ==================================================

category_importance = (
    importance_df.groupby("category")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

category_importance.columns = [
    "feature_category",
    "total_importance"
]

category_importance["percentage"] = (
    category_importance["total_importance"] * 100
)

category_file = (
    RESULTS_DIR
    / "feature_category_importance.csv"
)

category_importance.to_csv(
    category_file,
    index=False
)


# ==================================================
# 9. DISPLAY CATEGORY DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("IMPORTANCE BY FEATURE CATEGORY")
print("=" * 60)

print(
    category_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==================================================
# 10. CATEGORY IMPORTANCE PLOT
# ==================================================

plt.figure(figsize=(10, 6))

plt.bar(
    category_importance["feature_category"],
    category_importance["percentage"]
)

plt.xlabel("Feature Category")
plt.ylabel("Importance (%)")

plt.title(
    "Feature Category Importance - Random Forest"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

category_plot_file = (
    RESULTS_DIR
    / "feature_category_importance.png"
)

plt.savefig(
    category_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 11. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 60)

print(f"Total features analysed: {len(importance_df)}")

print("\nTop-ranked feature:")
print(importance_df.iloc[0]["feature"])

print("\nGenerated files:")
print(importance_file)
print(plot_file)
print(category_file)
print(category_plot_file)

print("\nAll analysis steps completed successfully.")
