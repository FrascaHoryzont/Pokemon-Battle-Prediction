"""Train the second version, report its cross-validated accuracy and write a submission.

Run from anywhere:  python second_version/train.py   (needs catboost: pip install catboost)
Output: second_version/submission.csv, in the exact format of sample_submission.csv.
Cross-validation takes several minutes because of CatBoost.
"""

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from features import build_tables, data_extr

SEED = 42
OUTPUT = Path(__file__).resolve().parent / "submission.csv"


def make_model():
    """Average of the probabilities of LightGBM and CatBoost.

    Neither beats the other on its own, but they make different mistakes, so the average is
    better. LightGBM settings come from a random search scored by log-loss.
    """
    lgbm = LGBMClassifier(n_estimators=700, learning_rate=0.0497, num_leaves=8, min_child_samples=20,
                          subsample=0.736, subsample_freq=1, colsample_bytree=0.0586, reg_lambda=1.0,
                          reg_alpha=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                             random_seed=SEED, verbose=0, thread_count=-1, allow_writing_files=False)
    return VotingClassifier([("lgbm", lgbm), ("cat", cat)], voting="soft")


def main():
    train, test = build_tables(data_extr.load_battles("train"), data_extr.load_battles("test"))
    X, y = train.drop(columns=["battle_id", "player_won"]), train["player_won"]

    # 5 folds, repeated with 3 different shuffles: 15 scores, so the ± is a reliable noise estimate.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scores = cross_val_score(make_model(), X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.2%} ± {scores.std():.2%}  (5 folds x 3 shuffles, {X.shape[1]} features)")

    model = make_model().fit(X, y)
    submission = pd.DataFrame({"battle_id": test["battle_id"],
                               "player_won": model.predict(test[X.columns]).astype(int)})
    sample = pd.read_csv(data_extr.REPO_ROOT / "sample_submission.csv")
    if submission["battle_id"].tolist() != sample["battle_id"].tolist():
        raise ValueError("battle_ids differ from sample_submission.csv")
    submission.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({submission['player_won'].mean():.1%} predicted P1 wins)")


if __name__ == "__main__":
    main()
