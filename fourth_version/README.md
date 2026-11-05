# Fourth version

Result: 69.11% ± 0.78% cross-validated accuracy (5 folds × 3 shuffles) and 69.11% aggregated OOF accuracy.

Models: Two-level Stacking Ensemble.
- Level 0 (Base Models): 15-fold bagged ensemble of XGBoost, LightGBM, and CatBoost.
- Level 1 (Meta-Learner): L2-regularized Logistic Regression trained on Out-Of-Fold (OOF) predicted probabilities from Level 0.

Key improvements over the third version:
- Replaced the arbitrary manual model weighting (60% XGB, 30% LGB, 10% CAT) with an automated, data-driven Stacking Meta-Learner.
- The Meta-Learner learns the optimal balance among models directly from cross-validation probabilities without manual tuning.
- Automatic decision boundary calibration via the learned intercept, removing the need for an empirical manual threshold.

Run: `python fourth_version/train.py`
