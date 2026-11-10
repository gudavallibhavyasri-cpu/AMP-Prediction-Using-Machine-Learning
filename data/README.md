
# APD3 Dataset

## Dataset Overview

The Antimicrobial Peptide Database (APD3) was used as
the source of peptide sequences for this project.

## Dataset Statistics

- Original FASTA records: 3,307
- Empty FASTA records: 1
- Non-empty FASTA records: 3,306
- Cleaned CSV sequences: 3,300
- Duplicate sequences in cleaned CSV: 0
- Invalid sequences in cleaned CSV: 0
- Sequence length range: 5–183 amino acids
- Average sequence length: 34.46 amino acids

## Sequence Filtering Observations

Six non-empty FASTA records were absent from the
cleaned CSV. All six had sequence lengths between
2 and 4 amino acids.

These observations are consistent with a minimum
sequence length filter of five amino acids.

The exact original filtering implementation has
not yet been independently verified.

## Data Quality

All 3,300 sequences in the cleaned CSV passed
the following checks:

- Standard amino acid alphabet validation
- Missing-value checks
- Duplicate sequence checks
- Sequence length verification

## Files

- `cleaned_apd3.csv`: Processed peptide sequences.
- `apd3.fasta`: Original FASTA dataset.

## Citation

Please consult the official APD3 database for
the appropriate dataset citation and usage terms
before redistributing the data.
