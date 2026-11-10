
from itertools import product
from collections import Counter

# ==========================================
# 1. STANDARD AMINO ACIDS
# ==========================================

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Generate all 400 possible dipeptides
DIPEPTIDES = [
    "".join(pair)
    for pair in product(AA_LIST, repeat=2)
]


# ==========================================
# 2. KYTE-DOOLITTLE HYDROPHOBICITY SCALE
# ==========================================

HYDRO = {
    'A': 1.8,
    'C': 2.5,
    'D': -3.5,
    'E': -3.5,
    'F': 2.8,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'K': -3.9,
    'L': 3.8,
    'M': 1.9,
    'N': -3.5,
    'P': -1.6,
    'Q': -3.5,
    'R': -4.5,
    'S': -0.8,
    'T': -0.7,
    'V': 4.2,
    'W': -0.9,
    'Y': -1.3
}


# ==========================================
# 3. FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_features(seq):

    # Standardise sequence
    seq = str(seq).strip().upper()

    # Validate sequence
    if not seq:
        raise ValueError("Empty peptide sequence.")

    invalid = set(seq) - set(AA_LIST)

    if invalid:
        raise ValueError(
            f"Invalid amino acid symbols: {invalid}"
        )

    length = len(seq)

    features = []

    # --------------------------------------
    # A. Amino Acid Composition (20)
    # --------------------------------------

    aac = [
        seq.count(aa) / length
        for aa in AA_LIST
    ]

    features.extend(aac)

    # --------------------------------------
    # B. Average Hydrophobicity (1)
    # --------------------------------------

    hydro = sum(
        HYDRO[aa] for aa in seq
    ) / length

    features.append(hydro)

    # --------------------------------------
    # C. Approximate Net Charge (1)
    # --------------------------------------

    charge = (
        seq.count('K')
        + seq.count('R')
        + seq.count('H')
        - seq.count('D')
        - seq.count('E')
    )

    features.append(charge)

    # --------------------------------------
    # D. Dipeptide Composition (400)
    # --------------------------------------

    dipeptides = [
        seq[i:i+2]
        for i in range(length - 1)
    ]

    counts = Counter(dipeptides)

    total_dp = len(dipeptides)

    dp_features = [
        counts[dp] / total_dp
        for dp in DIPEPTIDES
    ]

    features.extend(dp_features)

    # --------------------------------------
    # Final feature verification
    # --------------------------------------

    if len(features) != 422:
        raise ValueError(
            f"Expected 422 features, got {len(features)}"
        )

    return features


# ==========================================
# 4. FEATURE NAMES
# ==========================================

FEATURE_NAMES = (
    [f"AAC_{aa}" for aa in AA_LIST]
    + ["Average_Hydrophobicity"]
    + ["Approximate_Net_Charge"]
    + [f"DPC_{dp}" for dp in DIPEPTIDES]
)


# ==========================================
# 5. VERIFY FEATURE CONFIGURATION
# ==========================================

assert len(AA_LIST) == 20
assert len(DIPEPTIDES) == 400
assert len(FEATURE_NAMES) == 422
