"""Model factories"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# tuned with Optuna against the grouped 5-fold CV
def lgbm(seed: int = 0) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=595,
        learning_rate=0.022805,
        num_leaves=15,
        max_depth=5,
        min_child_samples=64,
        subsample=0.6912,
        colsample_bytree=0.5987,
        reg_alpha=1.8070,
        reg_lambda=0.014892,
        random_state=seed,
        verbosity=-1,
    )


# tuned with Optuna against the grouped 5-fold CV
def catboost(seed: int = 0) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=513,
        learning_rate=0.034633,
        depth=4,
        l2_leaf_reg=5.9461,
        random_seed=seed,
        verbose=False,
    )


def logistic_regression(seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


class ProbaBlend:
    """Equal-weight average of predict_proba across sub-models."""

    def __init__(self, model_fns):
        self.model_fns = model_fns

    def fit(self, X, y):
        self.models_ = [fn().fit(X, y) for fn in self.model_fns]
        return self

    def predict_proba(self, X):
        return sum(m.predict_proba(X) for m in self.models_) / len(self.models_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# final model: best CV of catboost/lgbm/all-3 blends
def lgbm_logreg_blend(seed: int = 0) -> ProbaBlend:
    return ProbaBlend([lambda: lgbm(seed), lambda: logistic_regression(seed)])
