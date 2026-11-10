from pathlib import Path
import pandas as pd
import re


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APD3_FILE = PROJECT_ROOT / "data" / "cleaned_apd3.csv"
DBAASP_FILE = PROJECT_ROOT / "data" / "raw" / "dbaasp.csv"
DRAMP_FILE = PROJECT_ROOT / "data" / "raw" / "dramp_general.xlsx"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 5


# ==================================================
# 2. SEQUENCE QUALITY ASSESSMENT
# ==================================================

def normalise_sequence(sequence):

    if pd.isna(sequence):
        return ""

    return str(sequence).strip().upper()


def assess_sequence(sequence):

    if not sequence:
        return "empty"

    if not re.fullmatch(r"[A-Z]+", sequence):
        return "non_letter_characters"

    if not set(sequence).issubset(STANDARD_AA):
        return "nonstandard_amino_acids"

    if len(sequence) < MIN_LENGTH:
        return "shorter_than_5"

    return "eligible"


# ==================================================
# 3. LOAD APD3
# ==================================================

print("\nLoading APD3...")

apd3 = pd.read_csv(APD3_FILE)

apd3_master = pd.DataFrame({

    "database": "APD3",

    "record_id": apd3["id"].astype(str),

    "sequence": apd3["sequence"].apply(normalise_sequence),

    "name": None,

    "activity": None,

    "target_organism": None,

    "synthesis_type": None,

    "complexity": "Monomer",

    "n_terminal_modification": None,

    "c_terminal_modification": None,

    "other_modifications": None,

    "stereochemistry": None,

    "source": None,

    "pubmed_id": None,

    "reference": None

})


# ==================================================
# 4. LOAD DBAASP
# ==================================================

print("Loading DBAASP...")

dbaasp = pd.read_csv(DBAASP_FILE)


dbaasp_master = pd.DataFrame({

    "database": "DBAASP",

    "record_id": dbaasp["ID"].astype(str),

    "sequence": dbaasp["SEQUENCE"].apply(normalise_sequence),

    "name": dbaasp["NAME"],

    "activity": None,

    "target_organism": (
        dbaasp["TARGET OBJECT"].fillna("")
        .astype(str)
    ),

    "synthesis_type": dbaasp["SYNTHESIS TYPE"],

    "complexity": dbaasp["COMPLEXITY"],

    "n_terminal_modification": dbaasp["N TERMINUS"],

    "c_terminal_modification": dbaasp["C TERMINUS"],

    "other_modifications": None,

    "stereochemistry": None,

    "source": None,

    "pubmed_id": None,

    "reference": None

})


# ==================================================
# 5. LOAD DRAMP
# ==================================================

print("Loading DRAMP...")

dramp = pd.read_excel(
    DRAMP_FILE,
    sheet_name="general_amps"
)


dramp_master = pd.DataFrame({

    "database": "DRAMP",

    "record_id": dramp["DRAMP_ID"].astype(str),

    "sequence": dramp["Sequence"].apply(normalise_sequence),

    "name": dramp["Name"],

    "activity": dramp["Activity"],

    "target_organism": dramp["Target_Organism"],

    "synthesis_type": None,

    "complexity": None,

    "n_terminal_modification": dramp["N-terminal_Modification"],

    "c_terminal_modification": dramp["C-terminal_Modification"],

    "other_modifications": dramp["Other_Modifications"],

    "stereochemistry": dramp["Stereochemistry"],

    "source": dramp["Source"],

    "pubmed_id": dramp["Pubmed_ID"],

    "reference": dramp["Reference"]

})


# ==================================================
# 6. COMBINE ALL RECORDS
# ==================================================

print("\nCombining datasets...")

master = pd.concat(
    [apd3_master, dbaasp_master, dramp_master],
    ignore_index=True
)

master["sequence_length"] = master["sequence"].str.len()

master["sequence_status"] = master["sequence"].apply(
    assess_sequence
)


# ==================================================
# 7. ADD DATABASE-SPECIFIC ELIGIBILITY
# ==================================================

def check_record_eligibility(row):

    if row["sequence_status"] != "eligible":
        return "excluded_sequence_quality"

    if (
        row["database"] == "DBAASP"
        and str(row["complexity"]).strip() != "Monomer"
    ):
        return "excluded_non_monomer"

    return "eligible"


master["initial_eligibility"] = master.apply(
    check_record_eligibility,
    axis=1
)


# ==================================================
# 8. EXACT SEQUENCE DUPLICATE INFORMATION
# ==================================================

print("Checking duplicate sequences...")

eligible = master[
    master["initial_eligibility"] == "eligible"
].copy()

sequence_database_counts = (
    eligible.groupby("sequence")["database"]
    .nunique()
)

sequence_record_counts = (
    eligible.groupby("sequence")
    .size()
)

master["number_of_source_databases"] = (
    master["sequence"].map(sequence_database_counts)
)

master["number_of_source_records"] = (
    master["sequence"].map(sequence_record_counts)
)

master["number_of_source_databases"] = (
    master["number_of_source_databases"].fillna(0).astype(int)
)

master["number_of_source_records"] = (
    master["number_of_source_records"].fillna(0).astype(int)
)


# ==================================================
# 9. SAVE COMPLETE MASTER DATASET
# ==================================================

master_file = PROCESSED_DIR / "amp_master_dataset.csv"

master.to_csv(
    master_file,
    index=False
)


# ==================================================
# 10. SAVE INITIAL ELIGIBLE DATASET
# ==================================================

eligible_records = master[
    master["initial_eligibility"] == "eligible"
].copy()

eligible_records.to_csv(
    PROCESSED_DIR / "amp_eligible_records.csv",
    index=False
)


# ==================================================
# 11. CREATE UNIQUE SEQUENCE DATASET
# ==================================================

unique_sequences = (
    eligible_records
    .sort_values(["sequence", "database", "record_id"])
    .drop_duplicates(subset=["sequence"], keep="first")
    .copy()
)

unique_sequences.to_csv(
    PROCESSED_DIR / "amp_unique_sequences.csv",
    index=False
)


# ==================================================
# 12. SAVE EXCLUDED RECORDS
# ==================================================

excluded_records = master[
    master["initial_eligibility"] != "eligible"
].copy()

excluded_records.to_csv(
    RESULTS_DIR / "amp_excluded_records.csv",
    index=False
)


# ==================================================
# 13. DATASET SUMMARY
# ==================================================

summary = (
    master.groupby(["database", "initial_eligibility"])
    .size()
    .unstack(fill_value=0)
)

summary["total_records"] = summary.sum(axis=1)

print("\n========== MASTER DATASET SUMMARY ==========")

print(summary.to_string())

summary.to_csv(
    RESULTS_DIR / "master_dataset_summary.csv"
)


# ==================================================
# 14. UNIQUE SEQUENCE SUMMARY
# ==================================================

print("\n========== UNIQUE SEQUENCE SUMMARY ==========")

print("Total source records:", len(master))

print("Initially eligible records:", len(eligible_records))

print("Unique eligible sequences:", len(unique_sequences))

print(
    "Eligible records containing repeated sequences:",
    len(eligible_records) - len(unique_sequences)
)


# ==================================================
# 15. COMPLETION
# ==================================================

print("\n==========================================")
print("MASTER DATASET BUILD COMPLETED")
print("==========================================")

print("\nGenerated files:")

print("data/processed/amp_master_dataset.csv")
print("data/processed/amp_eligible_records.csv")
print("data/processed/amp_unique_sequences.csv")

print("results/amp_excluded_records.csv")
print("results/master_dataset_summary.csv")

print("\nOriginal raw datasets were not modified.")