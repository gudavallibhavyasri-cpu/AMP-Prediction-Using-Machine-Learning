import pandas as pd
import numpy as np
import random
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load trained model
model = joblib.load("amp_model.pkl")

print(type(model))
print(model)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

HYDRO = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,
    'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
    'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
    'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}

def extract_features(seq):
    seq = seq.upper()
    length = len(seq)

    comp = [seq.count(aa)/length for aa in AA_LIST]

    hydro = sum(HYDRO.get(aa,0) for aa in seq)/length

    charge = (seq.count('K') + seq.count('R') + seq.count('H')) - \
             (seq.count('D') + seq.count('E'))

    return comp + [hydro, charge]

# Read AMP sequences
with open("validation_amps.txt") as f:
    amps = [line.strip() for line in f if line.strip()]

# Create scrambled negatives
negatives = []

for seq in amps:
    seq_list = list(seq)

    shuffled = seq_list[:]

    while ''.join(shuffled) == seq:
        random.shuffle(shuffled)

    negatives.append(''.join(shuffled))

print("AMPs loaded:", len(amps))
print("Negatives created:", len(negatives))

# Combine data
all_sequences = amps + negatives

true_labels = [1]*len(amps) + [0]*len(negatives)

# Extract features
X = np.array([extract_features(seq) for seq in all_sequences])

# Predict
probs = model.predict_proba(X)[:,1]

predictions = [1 if p >= 0.5 else 0 for p in probs]

# Metrics
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("\n===== VALIDATION RESULTS =====")
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))

print("\nConfusion Matrix")
print(confusion_matrix(true_labels, predictions))

# Save detailed table
results = pd.DataFrame({
    "sequence": all_sequences,
    "true_label": true_labels,
    "probability": probs,
    "prediction": predictions
})

results.to_csv("validation_results.csv", index=False)

print("\nSaved: validation_results.csv")