
import pandas as pd
from Bio import SeqIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = ROOT / "data" / "cleaned_apd3.csv"
FASTA_FILE = ROOT / "data" / "apd3.fasta"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load cleaned CSV
df = pd.read_csv(CSV_FILE)

csv_sequences = set(
    df["sequence"].astype(str).str.strip().str.upper()
)

# Load FASTA
records = list(SeqIO.parse(FASTA_FILE, "fasta"))

# Identify empty records
empty_records = [
    record for record in records
    if len(record.seq) == 0
]

# Identify sequences absent from CSV
missing_records = []

for record in records:
    sequence = str(record.seq).strip().upper()

    if sequence and sequence not in csv_sequences:
        missing_records.append({
            "record_id": record.id,
            "description": record.description,
            "sequence": sequence,
            "length": len(sequence)
        })

# Check repeated sequences in FASTA
fasta_sequences = [
    str(record.seq).strip().upper()
    for record in records
    if len(record.seq) > 0
]

print("TOTAL FASTA RECORDS:", len(records))
print("EMPTY RECORDS:", len(empty_records))
print("MISSING RECORDS:", len(missing_records))
print("DUPLICATE FASTA RECORDS:",
      len(fasta_sequences) - len(set(fasta_sequences)))

print("\nMISSING FASTA RECORDS:")

for item in missing_records:
    print("\nID:", item["record_id"])
    print("Description:", item["description"])
    print("Sequence:", item["sequence"])
    print("Length:", item["length"])

# Save missing records
missing_df = pd.DataFrame(missing_records)

missing_df.to_csv(
    RESULTS_DIR / "fasta_records_not_in_csv.csv",
    index=False
)

# Save empty record identifiers
with open(
    RESULTS_DIR / "empty_fasta_records.txt",
    "w",
    encoding="utf-8"
) as file:
    for record in empty_records:
        file.write(record.description + "\n")

print("\nInvestigation completed.")
print("Results saved in:", RESULTS_DIR)
