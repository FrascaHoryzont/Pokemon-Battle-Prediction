import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from data_parse import parse_jsonl_file

# create cross-validation folds and save them to a JSON file
def create_cv_folds():
    parsed_data = parse_jsonl_file(Path("data/train.jsonl"))

    # Prepare the data for StratifiedKFold
    battle_ids = [b['battle_id'] for b in parsed_data]
    y = [b['player_won'] for b in parsed_data]

    folds = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(battle_ids, y)):
        folds[f"fold_{fold_idx}"] = {
        "train": train_indices.tolist(),
        "val": val_indices.tolist()
        }
    with open("data/processed/cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=4)
    return folds

# Save the folds to a JSON file
def load_cv_folds(file_path="data/processed/cv_folds.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)