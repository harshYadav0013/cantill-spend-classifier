# Spend Classification Pipeline

A small Python pipeline I built for the Cantill.ai Data & ML challenge.

The goal was to combine procurement data from different systems, clean it, identify records that should not count as spend, and assign categories to the remaining records.

## What it does

- Loads and standardizes company data.
- Converts amounts into the reporting currency.
- Handles VAT where required.
- Identifies duplicates and excluded records.
- Generates consistent supplier and item keys.
- Classifies eligible records using the available taxonomy.
- Exports the processed data as CSV files.

## Project structure

```text
MyTask/
│
├── src/
│   ├── company_a.py
│   ├── company_b.py
│   ├── company_c.py
│   └── classifier.py
│
├── data/          # Input data (not included)
├── outputs/       # Generated CSV files
├── .gitignore
├── requirements.txt
├── README.md
├── APPROACH.md
└── DECISIONS.md
```

## Setup

Python 3.14 was used during development.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the provided input files inside their respective company folders under `data/`.

## Running

```bash
python src/company_a.py
python src/company_b.py
```

The processed files are saved in the `outputs/` directory.

Company C uses a separate zero-shot classification approach because labelled training data is not available.

## Notes

- Input company data is intentionally excluded from this repository.
- The classification model is a practical baseline, not a perfect classifier.
- Similarity scores from the zero-shot approach should not be interpreted as probabilities.