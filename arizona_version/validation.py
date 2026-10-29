"""Cross-validation helpers.

Grouped by P1 team composition: the same roster repeats a lot across rows,
so plain k-fold would leak a team's win rate across train/val. Grouping
keeps every row of a given team on one side of each fold.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def team_group_key(p1_team_details) -> str:
    return "|".join(sorted(mon["name"] for mon in p1_team_details))


def make_folds(df, n_splits: int = 5, seed: int = 0):
    groups = df["p1_team_details"].apply(team_group_key)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, df["player_won"], groups))


def cv_accuracy(model_fn, X, y, folds):
    """model_fn() returns a fresh, unfitted estimator. Returns (oof_preds, fold_accuracies)."""
    oof = np.zeros(len(y), dtype=int)
    fold_accs = []
    for train_idx, val_idx in folds:
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        oof[val_idx] = preds
        fold_accs.append((preds == y.iloc[val_idx].to_numpy()).mean())
    return oof, fold_accs
