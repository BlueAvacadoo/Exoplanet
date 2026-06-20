# TESS Catalog Pre-Sieve

This project cleans and categorizes TESS Input Catalog (TIC) CSV files before
more expensive light-curve analysis.

## Requirements

- Python 3.10+
- pandas
- NumPy

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the tests:

```bash
python3 -m unittest -v
```

## Analyze a TIC CSV

Create a complete analyzed CSV beside the input file. This preserves every
original column for targets that pass cleaning and appends category flags,
`selection_count`, and `selection_labels`:

```bash
python3 process_tess_targets.py \
  --input-csv "/path/to/tic_catalog.csv" \
  --complete-output \
  --categorize-all \
  --csv-chunk-size 50000
```

For the fastest and smallest output, retain only the columns needed by the
pre-sieve:

```bash
python3 process_tess_targets.py \
  --input-csv "/path/to/tic_catalog.csv" \
  --compact \
  --categorize-all
```

If `--output` is omitted, the result is saved beside the input as
`<input_name>_analyzed.csv`.

## Mandatory cleaning rules

1. Keep rows where `objType == STAR`.
2. Remove duplicate/artifact dispositions.
3. Require `contratio <= 0.10`.
4. Require `ruwe <= 1.40` when a RUWE column is available.

Categories are evaluated independently after cleaning. Missing data prevents a
target from matching only the categories that require that field.

Headered CSV columns may appear in any order. Headerless files must use the
standard 125-column TIC bulk-export order.

For remote MAST/Gaia acquisition rather than a local CSV, additionally install
`astroquery`.
