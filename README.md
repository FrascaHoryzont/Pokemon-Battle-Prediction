# Pokemon-Battle-Prediction

## Repository structure

The repository is organized as follows:

```text
Pokemon-Battle-Prediction/
├── data/
│   ├── train.jsonl              # Training battle records (local, ignored by Git)
│   ├── test.jsonl               # Test battle records (local, ignored by Git)
│   └── processed/
│       └── cv_folds.json        # Stratified cross-validation folds
├── src/
│   ├── data_parse.py            # JSONL data loading utilities
│   └── cv_splitter.py           # Five-fold stratified split generation/loading
├── submissions/
│   └── .gitkeep                 # Directory for generated CSV submissions
├── sample_submission.csv        # Expected submission format
├── type_chart_gen2.csv          # Generation 2 type effectiveness chart
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored datasets, caches, and local environments
├── LICENSE
└── README.md
```

### Directory responsibilities

- **`data/`** contains local datasets and processed artifacts. JSONL files and the
  entire directory are ignored by Git because the datasets can be large.
- **`src/`** contains the Python source code for parsing battle data and creating
  deterministic, stratified cross-validation folds.
- **`submissions/`** is reserved for CSV files generated for evaluation or
  competition submission.
- **`sample_submission.csv`** documents the required submission columns and format.
- **`type_chart_gen2.csv`** provides Generation 2 type matchup information for
  feature engineering.
- **`requirements.txt`** lists the Python packages required to run the project.

The local virtual environment should be created as `.venv/`. It is intentionally
ignored by Git and must not be committed.