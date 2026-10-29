"""raw jsonl -> features -> fit final model on all of train -> submission
"""

import json
from pathlib import Path

import pandas as pd

from features import build_features, build_species_lookup, build_type_chart_lookup
from models import lgbm_logreg_blend
from validation import cv_accuracy, make_folds

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "submission.csv"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    train = pd.DataFrame(load_jsonl(REPO_ROOT / "data" / "train.jsonl"))
    test = pd.DataFrame(load_jsonl(REPO_ROOT / "data" / "test.jsonl"))
    chart = build_type_chart_lookup(pd.read_csv(REPO_ROOT / "type_chart_gen2.csv"))
    species_lookup = build_species_lookup(train, test)

    X_train = build_features(train, species_lookup, chart).drop(columns=["battle_id"])
    X_test_full = build_features(test, species_lookup, chart)
    X_test = X_test_full.drop(columns=["battle_id"])
    assert list(X_train.columns) == list(X_test.columns)

    y = train["player_won"]
    folds = make_folds(train, n_splits=5, seed=0)
    _, accs = cv_accuracy(lgbm_logreg_blend, X_train, y, folds)
    print(f"grouped 5-fold CV accuracy: {sum(accs) / len(accs):.4f} (folds: {[round(a, 4) for a in accs]})")

    model = lgbm_logreg_blend().fit(X_train, y)
    preds = model.predict(X_test)

    submission = pd.DataFrame({"battle_id": X_test_full["battle_id"], "player_won": preds})
    sample = pd.read_csv(REPO_ROOT / "sample_submission.csv")
    assert set(submission["battle_id"]) == set(sample["battle_id"])

    submission.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} ({submission['player_won'].mean():.1%} predicted P1 wins)")


if __name__ == "__main__":
    main()
