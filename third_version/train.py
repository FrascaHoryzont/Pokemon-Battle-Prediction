"""Train Version 3, report 15-fold cross-validated accuracy, and write submission.

Pipeline:
1. Feature matrix assembly (577 rich Gen 2 domain, soft lookup, timeline & granular features)
2. Noise reduction: Top 200 most discriminative features selected via gradient boosting gain
3. Tri-model bagging ensemble across 15 folds (5-fold x 3 repeats):
   - XGBoost (60% weight): depth-wise splitting with quantile regularization
   - LightGBM (30% weight): leaf-wise histogram tree boosting
   - CatBoost (10% weight): symmetric oblivious trees
4. Calibrated decision threshold (tau=0.505) accounting for P1 vs P2 visibility asymmetry
5. 15-fold out-of-fold bagging for final test predictions

Run from anywhere:
    python third_version/train.py

Output:
    third_version/submission.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_extr  # noqa: E402
from third_version.features import build_tables  # noqa: E402

SEED = 42
N_TOP_FEATURES = 200
THRESHOLD = 0.505
OUTPUT = Path(__file__).resolve().parent / "submission.csv"


def make_model(fast: bool = False):
    """Ensemble averaging soft probabilities of XGBoost (60%), LightGBM (30%), and CatBoost (10%)."""
    if fast:
        lgbm = LGBMClassifier(n_estimators=10, num_leaves=8, random_state=SEED, n_jobs=1, verbose=-1)
        cat = CatBoostClassifier(iterations=10, depth=4, random_seed=SEED, verbose=0, thread_count=1, allow_writing_files=False)
        xgb = XGBClassifier(n_estimators=10, max_depth=3, random_state=SEED, n_jobs=1, eval_metric="logloss")
    else:
        xgb = XGBClassifier(
            n_estimators=550, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.3, reg_lambda=2.5, random_state=SEED, n_jobs=-1,
            eval_metric="logloss"
        )
        lgbm = LGBMClassifier(
            n_estimators=550, learning_rate=0.03, num_leaves=12, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=2.0,
            random_state=SEED, n_jobs=-1, verbose=-1
        )
        cat = CatBoostClassifier(
            iterations=650, learning_rate=0.03, depth=5, l2_leaf_reg=4,
            random_seed=SEED, verbose=0, thread_count=-1, allow_writing_files=False
        )

    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat)],
        voting="soft",
        weights=[0.60, 0.30, 0.10]
    )


def select_top_features(X_train: pd.DataFrame, y_train: pd.Series, n_top: int = N_TOP_FEATURES) -> list[str]:
    """Ranks and selects top n_top features using fast gradient boosting importance."""
    selector = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=4, colsample_bytree=0.3,
        subsample=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, eval_metric="logloss"
    )
    selector.fit(X_train, y_train)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    return importances.nlargest(n_top).index.tolist()


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    """Executes 15-fold cross-validation with OOF tracking and bagged test inference."""
    print(f"\nSelecting top {N_TOP_FEATURES} features from {X.shape[1]} candidates (pruning noise)...")
    selected_cols = select_top_features(X, y, n_top=N_TOP_FEATURES)
    print(f"Top 5 selected features: {', '.join(selected_cols[:5])}")

    X_sel = X[selected_cols]
    X_test_sel = X_test[selected_cols]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof_probs = np.zeros(len(X))
    test_probs = np.zeros(len(X_test))
    fold_accuracies = []

    n_folds = cv.get_n_splits()
    print(f"\nRunning {n_folds}-fold cross-validation with Tri-Model Ensemble & Test Bagging...")

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_sel, y)):
        X_tr, y_tr = X_sel.iloc[trn_idx], y.iloc[trn_idx]
        X_va, y_va = X_sel.iloc[val_idx], y.iloc[val_idx]

        # 1. XGBoost
        xgb = XGBClassifier(
            n_estimators=550, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.3, reg_lambda=2.5, random_state=SEED + fold, n_jobs=-1,
            eval_metric="logloss"
        )
        xgb.fit(X_tr, y_tr)
        p_xgb_val = xgb.predict_proba(X_va)[:, 1]
        p_xgb_test = xgb.predict_proba(X_test_sel)[:, 1]

        # 2. LightGBM
        lgb = LGBMClassifier(
            n_estimators=550, learning_rate=0.03, num_leaves=12, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=2.0,
            random_state=SEED + fold, n_jobs=-1, verbose=-1
        )
        lgb.fit(X_tr, y_tr)
        p_lgb_val = lgb.predict_proba(X_va)[:, 1]
        p_lgb_test = lgb.predict_proba(X_test_sel)[:, 1]

        # 3. CatBoost
        cat = CatBoostClassifier(
            iterations=650, learning_rate=0.03, depth=5, l2_leaf_reg=4,
            random_seed=SEED + fold, verbose=0, thread_count=-1, allow_writing_files=False
        )
        cat.fit(X_tr, y_tr)
        p_cat_val = cat.predict_proba(X_va)[:, 1]
        p_cat_test = cat.predict_proba(X_test_sel)[:, 1]

        # Weighted blend (60% XGB, 30% LGB, 10% CAT)
        val_blend = 0.60 * p_xgb_val + 0.30 * p_lgb_val + 0.10 * p_cat_val
        test_blend = 0.60 * p_xgb_test + 0.30 * p_lgb_test + 0.10 * p_cat_test

        oof_probs[val_idx] += val_blend / 3.0  # divided by 3 repeats
        test_probs += test_blend / float(n_folds)

        acc = accuracy_score(y_va, (val_blend >= THRESHOLD).astype(int))
        fold_accuracies.append(acc)
        print(f"  Fold {fold+1:2d}/{n_folds}: Accuracy = {acc:.2%}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    oof_acc = accuracy_score(y, (oof_probs >= THRESHOLD).astype(int))

    print(f"\n=======================================================")
    print(f"V3 Mean 15-Fold CV Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"V3 Aggregated OOF Accuracy:  {oof_acc:.2%} (Threshold={THRESHOLD})")
    print(f"=======================================================\n")

    return test_probs


def main():
    print("Loading raw training and test battles...")
    train_battles = data_extr.load_battles("train")
    test_battles = data_extr.load_battles("test")

    print("Building feature matrices for Version 3...")
    train, test = build_tables(train_battles, test_battles)

    X = train.drop(columns=["battle_id", "player_won"])
    y = train["player_won"]
    X_test = test.drop(columns=["battle_id", "player_won"], errors="ignore")

    print(f"\nInitial feature matrix: {X.shape[0]} battles × {X.shape[1]} rich candidate features")

    test_probs = train_and_evaluate(X, y, X_test)

    predictions = (test_probs >= THRESHOLD).astype(int)

    submission = pd.DataFrame({
        "battle_id": test["battle_id"],
        "player_won": predictions
    })

    sample = pd.read_csv(REPO_ROOT / "sample_submission.csv")
    if submission["battle_id"].tolist() != sample["battle_id"].tolist():
        raise ValueError("battle_ids do not match sample_submission.csv!")

    submission.to_csv(OUTPUT, index=False)
    p1_win_rate = submission["player_won"].mean()
    print(f"Saved submission to {OUTPUT} ({p1_win_rate:.1%} predicted P1 wins)")


if __name__ == "__main__":
    main()
