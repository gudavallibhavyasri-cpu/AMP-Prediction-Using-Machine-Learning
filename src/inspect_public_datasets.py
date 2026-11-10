
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================
# 2. INSPECT CSV FILES
# ============================================================

def inspect_csv(file_path):

    print("\n" + "=" * 70)
    print(f"CSV FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 3. INSPECT EXCEL FILES
# ============================================================

def inspect_excel(file_path):

    print("\n" + "=" * 70)
    print(f"EXCEL FILE: {file_path.name}")
    print("=" * 70)

    excel = pd.ExcelFile(file_path)

    print("Sheets:", excel.sheet_names)

    for sheet in excel.sheet_names:

        df = pd.read_excel(file_path, sheet_name=sheet)

        print("\nSheet:", sheet)
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nColumn names:")
        for column in df.columns:
            print("-", column)

        print("\nFirst 5 rows:")
        print(df.head().to_string())

        print("\nMissing values:")
        print(df.isnull().sum().to_string())


# ============================================================
# 4. INSPECT FASTA FILES
# ============================================================

def inspect_fasta(file_path):

    print("\n" + "=" * 70)
    print(f"FASTA FILE: {file_path.name}")
    print("=" * 70)

    total_records = 0
    empty_sequences = 0
    unique_sequences = set()
    lengths = []
    unusual_records = []
    first_records = []

    current_sequence = []
    current_id = None
    current_header = None

    def process_record(record_id, sequence, header):

        nonlocal empty_sequences

        sequence = "".join(sequence.split()).upper()

        if not sequence:
            empty_sequences += 1
            return

        unique_sequences.add(sequence)
        lengths.append(len(sequence))

        unusual = set(sequence) - STANDARD_AA

        if unusual:
            unusual_records.append(
                (record_id, sequence, sorted(unusual))
            )

        if len(first_records) < 5:
            first_records.append((record_id, sequence))

    # Read file as bytes to avoid Windows encoding errors
    with open(file_path, "rb") as handle:

        for line_number, raw_line in enumerate(handle, start=1):

            line = raw_line.decode(
                "utf-8-sig",
                errors="replace"
            ).strip()

            if not line:
                continue

            if line.startswith(">"):

                if current_id is not None:
                    process_record(
                        current_id,
                        "".join(current_sequence),
                        current_header
                    )

                total_records += 1

                current_header = line[1:]
                current_id = current_header.split()[0]
                current_sequence = []

            else:
                if current_id is not None:
                    current_sequence.append(line)

        # Process final record
        if current_id is not None:
            process_record(
                current_id,
                "".join(current_sequence),
                current_header
            )

    print("Total records:", total_records)
    print("Empty sequences:", empty_sequences)
    print("Unique non-empty sequences:", len(unique_sequences))

    if lengths:

        print("Minimum length:", min(lengths))
        print("Maximum length:", max(lengths))

        print(
            "Average length:",
            round(sum(lengths) / len(lengths), 2)
        )

    print("\nFirst 5 records:")

    for record_id, sequence in first_records:
        print("ID:", record_id)
        print("Sequence:", sequence)

    print("\nRecords containing non-standard characters:",
          len(unusual_records))

    if unusual_records:

        print("\nFirst 10 unusual records:")

        for record_id, sequence, unusual in unusual_records[:10]:

            print("ID:", record_id)
            print("Sequence:", sequence)
            print("Unusual characters:", unusual)
            print("-" * 40)


# ============================================================
# 5. INSPECT TEXT FILES
# ============================================================

def inspect_text(file_path):

    print("\n" + "=" * 70)
    print(f"TEXT FILE: {file_path.name}")
    print("=" * 70)

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        print("First 15 lines:\n")

        for line_number, line in enumerate(file):

            if line_number >= 15:
                break

            print(line.rstrip())


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def main():

    print("\nPUBLIC AMP DATASET INSPECTION")
    print("=" * 70)

    print("Project directory:", PROJECT_DIR)
    print("Raw data directory:", RAW_DIR)

    # --------------------------------------------------------
    # DBAASP FILES
    # --------------------------------------------------------

    print("\n\n*************** DBAASP DATASET ***************")

    dbaasp_csv = RAW_DIR / "dbaasp.csv"
    dbaasp_fasta = RAW_DIR / "dbaasp.fasta"

    if dbaasp_csv.exists():
        inspect_csv(dbaasp_csv)
    else:
        print("Missing file:", dbaasp_csv.name)

    if dbaasp_fasta.exists():
        inspect_fasta(dbaasp_fasta)
    else:
        print("Missing file:", dbaasp_fasta.name)

    # --------------------------------------------------------
    # DRAMP FILES
    # --------------------------------------------------------

    print("\n\n**************** DRAMP DATASET ****************")

    dramp_excel = RAW_DIR / "dramp_general.xlsx"
    dramp_fasta = RAW_DIR / "dramp_general.fasta"
    dramp_text = RAW_DIR / "dramp_general.txt"

    if dramp_excel.exists():
        inspect_excel(dramp_excel)
    else:
        print("Missing file:", dramp_excel.name)

    if dramp_fasta.exists():
        inspect_fasta(dramp_fasta)
    else:
        print("Missing file:", dramp_fasta.name)

    if dramp_text.exists():
        inspect_text(dramp_text)
    else:
        print("Missing file:", dramp_text.name)

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


# ============================================================
# 7. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
