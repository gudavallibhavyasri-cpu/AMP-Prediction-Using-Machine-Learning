
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from collections import Counter

# ==========================================
# 1. PROJECT PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / "cleaned_apd3.csv"
FASTA_FILE = DATA_DIR / "apd3.fasta"

AA_LIST = set("ACDEFGHIKLMNPQRSTVWY")

print("=" * 55)
print("APD3 DATASET QUALITY AUDIT")
print("=" * 55)


# ==========================================
# 2. LOAD CLEANED CSV DATASET
# ==========================================

df = pd.read_csv(CSV_FILE)

print("\n1. CSV INFORMATION")
print("Total sequences:", len(df))
print("Columns:", df.columns.tolist())


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\n2. MISSING VALUES")
print(df.isnull().sum())


# ==========================================
# 4. CHECK SEQUENCE VALIDITY
# ==========================================

sequences = df["sequence"].astype(str).str.upper()

invalid = sequences[
    ~sequences.apply(
        lambda seq: len(seq) > 0
        and set(seq).issubset(AA_LIST)
    )
]

print("\n3. SEQUENCE VALIDITY")
print("Valid sequences:", len(sequences) - len(invalid))
print("Invalid sequences:", len(invalid))


# ==========================================
# 5. CHECK DUPLICATE SEQUENCES
# ==========================================

duplicates = sequences.duplicated().sum()

print("\n4. DUPLICATE ANALYSIS")
print("Duplicate sequences:", duplicates)


# ==========================================
# 6. VERIFY SEQUENCE LENGTHS
# ==========================================

calculated_lengths = sequences.str.len()

length_mismatches = (
    df["length"].values != calculated_lengths.values
).sum()

print("\n5. LENGTH VERIFICATION")
print("Length mismatches:", length_mismatches)
print("Minimum length:", calculated_lengths.min())
print("Maximum length:", calculated_lengths.max())
print("Average length:", round(calculated_lengths.mean(), 2))


# ==========================================
# 7. AMINO ACID DISTRIBUTION
# ==========================================

aa_counts = Counter("".join(sequences))

print("\n6. AMINO ACID DISTRIBUTION")

for aa in sorted(AA_LIST):
    print(aa, ":", aa_counts.get(aa, 0))


# ==========================================
# 8. LOAD ORIGINAL FASTA FILE
# ==========================================

records = list(SeqIO.parse(FASTA_FILE, "fasta"))

empty_records = [
    record for record in records
    if len(record.seq) == 0
]

valid_records = [
    record for record in records
    if len(record.seq) > 0
]

print("\n7. FASTA INFORMATION")
print("Total FASTA records:", len(records))
print("Empty FASTA records:", len(empty_records))
print("Non-empty FASTA records:", len(valid_records))


# ==========================================
# 9. COMPARE CSV AND FASTA
# ==========================================

csv_set = set(sequences)

fasta_set = {
    str(record.seq).upper()
    for record in valid_records
}

print("\n8. CSV AND FASTA COMPARISON")
print("Sequences in CSV:", len(csv_set))
print("Sequences in FASTA:", len(fasta_set))

print(
    "CSV sequences missing from FASTA:",
    len(csv_set - fasta_set)
)

print(
    "FASTA sequences missing from CSV:",
    len(fasta_set - csv_set)
)


# ==========================================
# 10. SAVE AUDIT REPORT
# ==========================================

report = f"""
APD3 DATASET QUALITY AUDIT

CSV sequences: {len(df)}
Invalid sequences: {len(invalid)}
Duplicate sequences: {duplicates}
Length mismatches: {length_mismatches}

Minimum sequence length: {calculated_lengths.min()}
Maximum sequence length: {calculated_lengths.max()}
Average sequence length: {calculated_lengths.mean():.2f}

FASTA records: {len(records)}
Empty FASTA records: {len(empty_records)}
Non-empty FASTA records: {len(valid_records)}

CSV sequences missing from FASTA: {len(csv_set - fasta_set)}
FASTA sequences missing from CSV: {len(fasta_set - csv_set)}
"""

report_file = RESULTS_DIR / "dataset_audit.txt"

report_file.write_text(report, encoding="utf-8")

print("\n" + "=" * 55)
print("AUDIT COMPLETED")
print("Report saved to:", report_file)
print("=" * 55)
