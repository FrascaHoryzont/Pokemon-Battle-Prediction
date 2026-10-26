from pathlib import Path
from cv_splitter import load_cv_folds
from data_parse import load_type_chart, parse_jsonl_file
from features_endstate import extract_endstate_features
from features_static import extract_static_features
from features_timeline import extract_timeline_features
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Check if LightGBM is available for use, otherwise fallback to HistGradientBoostingClassifier.
try:
  from lightgbm import LGBMClassifier

  HAS_LGBM = True
except ImportError:
  HAS_LGBM = False

# Build a feature matrix by combining end state, static, and timeline features for the given battles.
def build_feature_matrix(battles, type_chart):
  df_end = extract_endstate_features(battles)
  df_stat = extract_static_features(battles, type_chart)
  df_time = extract_timeline_features(battles)
  return pd.concat([df_end, df_stat, df_time], axis=1)

# Main function to execute the training pipeline, including feature extraction, model training, cross-validation, and submission file generation.
def main():
  train_battles = parse_jsonl_file(Path("data/train.jsonl"))
  test_battles = parse_jsonl_file(Path("data/test.jsonl"))
  type_chart = load_type_chart(Path("type_chart_gen2.csv"))
  folds = load_cv_folds("data/processed/cv_folds.json")

  print("Extracting features...")
  X_train = build_feature_matrix(train_battles, type_chart)
  y_train = pd.Series(
      [int(b["player_won"]) for b in train_battles],
      index=X_train.index,
      name="player_won",
  )
  X_test = build_feature_matrix(test_battles, type_chart)

  oof_preds = np.zeros(len(X_train))
  test_preds_proba = np.zeros(len(X_test))
  fold_accuracies = []

  for fold_name, indices in folds.items():
    tr_idx, val_idx = indices["train"], indices["val"]

    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    if HAS_LGBM:
      model = LGBMClassifier(
          n_estimators=300,
          learning_rate=0.05,
          num_leaves=31,
          random_state=42,
          verbose=-1,
      )
    else:
      model = HistGradientBoostingClassifier(
          max_iter=300, learning_rate=0.05, random_state=42
      )

    model.fit(X_tr, y_tr)

    val_proba = model.predict_proba(X_va)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    oof_preds[val_idx] = val_pred

    acc = accuracy_score(y_va, val_pred)
    fold_accuracies.append(acc)
    print(f"{fold_name} Accuracy: {acc:.4f}")

    test_preds_proba += model.predict_proba(X_test)[:, 1] / len(folds)

  oof_acc = accuracy_score(y_train, oof_preds)
  print(
      f"\nOOF CV Accuracy Totale: {oof_acc:.4f} (Std:"
      f" {np.std(fold_accuracies):.4f})"
  )

  # Save the submission file based on the test predictions and align it with the sample submission format.
  sample_sub = pd.read_csv(Path("sample_submission.csv"))
  sub = pd.DataFrame({
      "battle_id": X_test.index,
      "player_won": (test_preds_proba >= 0.5).astype(int),
  })

  # Verify that the submission aligns with the sample submission
  sub = sample_sub[["battle_id"]].merge(sub, on="battle_id", how="left")
  sub.to_csv(Path("submission_baseline.csv"), index=False)
  print("File saved in the root: submission_baseline.csv")


if __name__ == "__main__":
  main()