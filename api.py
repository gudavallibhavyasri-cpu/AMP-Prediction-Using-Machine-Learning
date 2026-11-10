
from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "models" / "random_forest_optimized_compressed.pkl"

sys.path.insert(0, str(SRC_DIR))

from feature_extraction import extract_features, FEATURE_NAMES

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Check expected feature count
if model.n_features_in_ != len(FEATURE_NAMES):
    raise ValueError(
        f"Model expects {model.n_features_in_} features, "
        f"but feature extractor provides {len(FEATURE_NAMES)}."
    )

app = FastAPI(
    title="Antimicrobial Peptide Prediction API",
    description="Machine learning baseline for AMP sequence classification.",
    version="1.0.0"
)


class PeptideInput(BaseModel):
    sequence: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Peptide sequence using standard amino-acid letters."
    )


@app.get("/")
def home():
    return {
        "message": "AMP Prediction API is running",
        "model": "Optimised Random Forest",
        "features": len(FEATURE_NAMES)
    }


@app.post("/predict")
def predict(data: PeptideInput):

    sequence = data.sequence.strip().upper()

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    if not sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty.")

    invalid = set(sequence) - valid_amino_acids

    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid amino-acid characters: {', '.join(sorted(invalid))}"
        )

    try:
        feature_values = extract_features(sequence)

        features_df = pd.DataFrame(
            [feature_values],
            columns=FEATURE_NAMES
        )

        probabilities = model.predict_proba(features_df)[0]
        classes = list(model.classes_)

        positive_index = classes.index(1)
        score = float(probabilities[positive_index])

        prediction = int(model.predict(features_df)[0])

        return {
            "sequence": sequence,
            "prediction": "AMP-like" if prediction == 1 else "Non-AMP-like",
            "amp_class_score": round(score, 4),
            "model": "Optimised Random Forest",
            "feature_count": len(FEATURE_NAMES),
            "interpretation": (
                "Score represents similarity to the database AMP class "
                "versus synthetic shuffled controls. It is not an "
                "experimentally validated probability of antimicrobial activity."
            )
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
