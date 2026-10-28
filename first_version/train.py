"""Train the first version, report its cross-validated accuracy and write a submission.

Run from anywhere:  python first_version/train.py
Output: first_version/submission.csv, in the exact format of sample_submission.csv.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_table, data_extr

SEED = 42
OUTPUT = Path(__file__).resolve().parent / "submission.csv"


def make_model():
    """Scaling + logistic regression.

    On these features it matched or beat random forest and LightGBM (all ~67%),
    and C=0.1 scored best among 0.01, 0.1 and 1. Scaling is inside the
    pipeline, so in cross-validation it is refitted on each training part.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=SEED))


def main():
    train = build_table(data_extr.load_battles("train"))
    test = build_table(data_extr.load_battles("test"))
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
