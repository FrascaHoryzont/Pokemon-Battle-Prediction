# Pokémon Battle Prediction — FDS Kaggle Challenge (Gen 2 OU)

Official team repository for the **FDS Kaggle Competition** (*Fundamentals/Foundations of Data Science* — Sapienza University of Rome).

## Project Goal

The objective of this challenge is to train a **supervised binary classification** model capable of predicting the final outcome of a Pokémon battle in the *Generation 2 OverUsed (OU)* competitive format (`player_won`: `1` if Player 1 wins, `0` if Player 1 loses), observing **only the initial team composition and the first 30 active turns** of the match.

### Competition Constraints
* **Evaluation Metric:** Accuracy.
* **Allowed Algorithms:** Classical Machine Learning models only (e.g., *Gradient Boosting*, *Random Forest*, *Logistic Regression*). **Neural networks are strictly prohibited.**
* **Allowed Data:** Official competition files only (`train.jsonl`, `test.jsonl`, `type_chart_gen2.csv`).

## Repository Structure

The project is organized into modular scripts to allow parallel development across 4 team members without Git conflicts. CSV files are located in the project **root**, `.jsonl` datasets are inside `data/`, and Python source files are inside `src/`:

```text
progetto_pokemon/
├── README.md
├── type_chart_gen2.csv
├── sample_submission.csv
├── submission_baseline.csv
├── data/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── processed/
│       └── cv_folds.json
└── src/
    ├── __init__.py
    ├── data_parse.py
    ├── cv_splitter.py
    ├── features_endstate.py
    ├── features_static.py
    ├── features_timeline.py
    └── train_pipeline.py
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

## Setup and Usage

### Prerequisites
Install Python 3.9+ and the required dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm tqdm
```

*(Note: If `lightgbm` is not installed, the pipeline automatically falls back to `HistGradientBoostingClassifier` from `scikit-learn`).*

### Data Placement
1. Place `type_chart_gen2.csv` and `sample_submission.csv` in the project **root**.
2. Create the `data/` folder in the root directory and place `train.jsonl` and `test.jsonl` inside it.

### Running the Pipeline
Run all commands from the project **root** directory (`Pokemon-Battle-Prediction/`):

**Step 1: Generate Cross-Validation Folds** (run once)
```bash
python src/cv_splitter.py
```
Reads `data/train.jsonl` and generates `data/processed/cv_folds.json` using a **5-Fold Stratified Cross-Validation** (`random_state=42`).

**Step 2: Feature Extraction, CV Evaluation, and Submission Generation**
```bash
python src/train_pipeline.py
```
This script:
1. Loads the training set, test set, and type chart.
2. Extracts and concatenates features horizontally from `features_endstate.py`, `features_static.py`, and `features_timeline.py`.
3. Trains the model across the 5 folds, printing individual fold accuracies and the overall **Out-Of-Fold (OOF) Accuracy**.
4. Generates `submission_baseline.csv` in the project root.

## Feature Engineering & Team Workflow

Each feature module takes the parsed `battles` list as input and returns a `pd.DataFrame` indexed by `battle_id`.

| Module | Role | Extracted Features |
| :--- | :--- | :--- |
| **`src/features_endstate.py`** | **Member 1** *(End-State)* | Captures the battle state at turn 30: estimated total remaining HP across both teams (assuming 100% HP for unrevealed Pokémon), fainted Pokémon counts (`fnt` or `hp_pct == 0`), number of revealed Player 2 Pokémon, and active status conditions. |
| **`src/features_static.py`** | **Member 2** *(Pre-Battle & Types)* | Extracts static pre-battle signals: mean, max, and standard deviation of Player 1's base stats (`hp`, `atk`, `def`, `spa`, `spd`, `spe`), stat and speed differentials between Player 1's lead and Player 2's lead, `leftovers` count, and type matchup multipliers via `type_chart_gen2.csv`. |
| **`src/features_timeline.py`** | **Member 3** *(Timeline Dynamics)* | Summarizes turn-by-turn dynamics (turns 1–30): switches/inactive turns (`move_details == null`), average move `base_power`, status move and `rest` usage, accumulated stat `boosts`, and HP differential trends between the first 10 turns and the last 10 turns (*momentum*). |
| **`src/train_pipeline.py`** | **Member 4** *(Modeling & CV)* | Handles feature matrix assembly (`build_feature_matrix`), Out-Of-Fold cross-validation, feature selection, hyperparameter tuning, and model ensembling. |

### Adding New Features
1. Open only your assigned module in `src/`.
2. Compute the new feature inside the `for b in battles:` loop and add the key-value pair to `rows.append({...})`.
3. Run `python src/train_pipeline.py` to check if the **OOF CV Accuracy** improves.

## Model & Validation Strategy

### Gradient Boosted Decision Trees (GBDT)
The pipeline uses **LightGBM (`LGBMClassifier`)** (with **`HistGradientBoostingClassifier`** as a fallback) for three main reasons:
1. **Non-linear feature interactions:** Captures conditional relationships between remaining HP, accumulated stat boosts, and speed tiers.
2. **Mixed feature types and scales:** Natively handles continuous percentages (`hp_pct`), discrete counts, and base stats without requiring feature scaling.
3. **Fast K-Fold iteration:** Histogram-based splitting trains 5 folds over 10,000 battles in a few seconds.

### Validation and Inference (Fold Averaging)
* **Anti-leakage local validation:** Feature selection and hyperparameter tuning rely exclusively on the **Out-Of-Fold (OOF) Accuracy** computed on `data/processed/cv_folds.json`.
* **Test inference:** Predictions on `data/test.jsonl` are generated by averaging the predicted probabilities (`predict_proba`) across all 5 fold models (*Soft Voting / Fold Averaging*) before applying the `>= 0.5` classification threshold.