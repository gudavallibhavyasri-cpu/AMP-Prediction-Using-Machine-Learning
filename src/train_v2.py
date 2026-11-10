from itertools import product
from collections import Counter
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# Amino acids
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# All possible dipeptides (20 × 20 = 400)
DIPEPTIDES = [''.join(dp) for dp in product(AA_LIST, repeat=2)]

# Hydrophobicity scale
HYDRO = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,
    'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
    'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
    'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}

def extract_features(seq):

    length = len(seq)

    # AAC
    comp = [seq.count(aa)/length for aa in AA_LIST]

    # Hydrophobicity
    hydro = sum(HYDRO.get(aa,0) for aa in seq)/length

    # Charge
    charge = (
        seq.count('K') +
        seq.count('R') +
        seq.count('H')
        -
        seq.count('D')
        -
        seq.count('E')
    )

    # Dipeptide composition
    dipeptides = [seq[i:i+2] for i in range(len(seq)-1)]

    counts = Counter(dipeptides)

    total_dp = max(len(dipeptides),1)

    dp_features = [
        counts[dp]/total_dp
        for dp in DIPEPTIDES
    ]

    return comp + [hydro, charge] + dp_features


# Load APD3
df = pd.read_csv("cleaned_apd3.csv")

df["label"] = 1

# Generate negatives
def random_seq(length):
    return ''.join(np.random.choice(AA_LIST, length))

neg_sequences = [
    random_seq(np.random.randint(5,50))
    for _ in range(len(df))
]

neg_df = pd.DataFrame({
    "sequence": neg_sequences,
    "label": 0
})

# Combine
data = pd.concat(
    [df[["sequence","label"]], neg_df],
    ignore_index=True
)

# Features
X = np.array([
    extract_features(seq)
    for seq in data["sequence"]
])

y = data["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Calibration
model = CalibratedClassifierCV(
    rf,
    cv=3
)

model.fit(X_train, y_train)

# AUC
y_pred = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_pred)

print("AUC:", auc)

# Save
joblib.dump(model, "amp_model_v2.pkl")

print("Saved amp_model_v2.pkl")