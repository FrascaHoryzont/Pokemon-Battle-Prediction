"""Train Version 4: Stacking Ensemble with Logistic Regression Meta-Learner.

Pipeline:
1. Feature matrix assembly (565 rich Gen 2 domain, soft lookup, timeline & granular features)
2. Noise reduction: Top 200 most discriminative features selected via gradient boosting gain
3. Level 0 Base Models across 15 folds (5-fold x 3 repeats):
   - XGBoost: depth-wise splitting with quantile regularization
   - LightGBM: leaf-wise histogram tree boosting
   - CatBoost: symmetric oblivious trees
4. Level 1 Stacking Meta-Learner:
   - Logistic Regression trained on Out-Of-Fold base model predictions
   - Automatically learns optimal model blend weights and calibration intercept
5. 15-fold out-of-fold bagging for final test predictions

Run from anywhere:
    python fourth_version/train.py

Output:
    fourth_version/submission.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_extr  # noqa: E402
from fourth_version.features import build_tables  # noqa: E402

SEED = 42
N_TOP_FEATURES = 200
OUTPUT = Path(__file__).resolve().parent / "submission.csv"


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
    """Executes 15-fold cross-validation with Level 0 OOF tracking and Level 1 Logistic Meta-Learner."""
    print(f"\nSelecting top {N_TOP_FEATURES} features from {X.shape[1]} candidates (pruning noise)...")
    selected_cols = select_top_features(X, y, n_top=N_TOP_FEATURES)
    print(f"Top 5 selected features: {', '.join(selected_cols[:5])}")

    X_sel = X[selected_cols]
    X_test_sel = X_test[selected_cols]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    n_folds = cv.get_n_splits()

    # Level 0 OOF probability containers
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    print(f"\n[Level 0] Training 3 Base Models (XGBoost, LightGBM, CatBoost) across {n_folds} folds...")

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

        # Accumulate out-of-fold (averaged across 3 repeats)
        oof_xgb[val_idx] += p_xgb_val / 3.0
        oof_lgb[val_idx] += p_lgb_val / 3.0
        oof_cat[val_idx] += p_cat_val / 3.0

        # Accumulate test predictions across 15 folds
        test_xgb += p_xgb_test / float(n_folds)
        test_lgb += p_lgb_test / float(n_folds)
        test_cat += p_cat_test / float(n_folds)

        acc_xgb = accuracy_score(y_va, (p_xgb_val >= 0.5).astype(int))
        acc_lgb = accuracy_score(y_va, (p_lgb_val >= 0.5).astype(int))
        acc_cat = accuracy_score(y_va, (p_cat_val >= 0.5).astype(int))
        print(f"  Fold {fold+1:2d}/{n_folds}: XGB={acc_xgb:.2%} | LGB={acc_lgb:.2%} | CAT={acc_cat:.2%}")

    # Baseline comparison: manual fixed blend (V3 style: 60/30/10 with tau=0.505)
    v3_oof_blend = 0.60 * oof_xgb + 0.30 * oof_lgb + 0.10 * oof_cat
    v3_oof_acc = accuracy_score(y, (v3_oof_blend >= 0.505).astype(int))

    print(f"\n[Level 0 Summary]")
    print(f"  Single Model OOF: XGB={accuracy_score(y, (oof_xgb >= 0.5).astype(int)):.2%} | LGB={accuracy_score(y, (oof_lgb >= 0.5).astype(int)):.2%} | CAT={accuracy_score(y, (oof_cat >= 0.5).astype(int)):.2%}")
    print(f"  V3 Manual Fixed Blend OOF (60/30/10, tau=0.505): {v3_oof_acc:.2%}")

    # Level 1: Logistic Regression Meta-Learner
    print(f"\n[Level 1] Training Logistic Regression Meta-Learner on Out-Of-Fold probabilities...")
    X_meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
    X_meta_test = np.column_stack([test_xgb, test_lgb, test_cat])

    # Cross-validate the Meta-Learner to evaluate generalization without leakage
    meta_oof_probs = np.zeros(len(y))
    meta_test_probs = np.zeros(len(X_test))
    meta_fold_accuracies = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_meta_train, y)):
        meta_lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED + fold)
        meta_lr.fit(X_meta_train[trn_idx], y.iloc[trn_idx])

        p_meta_val = meta_lr.predict_proba(X_meta_train[val_idx])[:, 1]
        p_meta_test = meta_lr.predict_proba(X_meta_test)[:, 1]

        meta_oof_probs[val_idx] += p_meta_val / 3.0
        meta_test_probs += p_meta_test / float(n_folds)

        acc = accuracy_score(y.iloc[val_idx], (p_meta_val >= 0.5).astype(int))
        meta_fold_accuracies.append(acc)

    meta_mean_acc = np.mean(meta_fold_accuracies)
    meta_std_acc = np.std(meta_fold_accuracies)
    meta_oof_acc = accuracy_score(y, (meta_oof_probs >= 0.5).astype(int))

    # Inspect the final meta-learner weights
    final_meta = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    final_meta.fit(X_meta_train, y)
    w_xgb, w_lgb, w_cat = final_meta.coef_[0]
    intercept = final_meta.intercept_[0]

    print(f"\n=======================================================")
    print(f"Learned Meta-Weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}, CAT={w_cat:.3f}")
    print(f"Learned Intercept (Bias): {intercept:.3f}")
    print(f"-------------------------------------------------------")
    print(f"V4 Stacking Mean 15-Fold CV Accuracy: {meta_mean_acc:.2%} ± {meta_std_acc:.2%}")
    print(f"V4 Stacking Aggregated OOF Accuracy:  {meta_oof_acc:.2%}")
    print(f"Delta vs V3 Fixed Blend:              {meta_oof_acc - v3_oof_acc:+.2%}")
    print(f"=======================================================\n")

    return meta_test_probs


def main():
    print("Loading raw training and test battles...")
    train_battles = data_extr.load_battles("train")
    test_battles = data_extr.load_battles("test")

    print("Building feature matrices for Version 4...")
    train, test = build_tables(train_battles, test_battles)

    X = train.drop(columns=["battle_id", "player_won"])
    y = train["player_won"]
    X_test = test.drop(columns=["battle_id", "player_won"], errors="ignore")

    print(f"\nInitial feature matrix: {X.shape[0]} battles × {X.shape[1]} rich candidate features")

    test_probs = train_and_evaluate(X, y, X_test)

    predictions = (test_probs >= 0.5).astype(int)

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
