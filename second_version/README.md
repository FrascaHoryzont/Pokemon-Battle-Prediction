# Second version

Result: 68.64% ± 0.86% cross-validated accuracy (5 folds × 3 shuffles), up from 66.88% for the first version.

Models: average of the probabilities of LightGBM (small trees, 8 leaves) and CatBoost (depth 6).

Features (470):
- The first version's: fainted Pokémon and team HP left (with P2's unrecorded knock-outs counted), statuses, active Pokémon, reserves, team strength.
- Spikes still up on each side.
- Which species are still alive and which moves each side has used.
- Survivors: type and speed matchups, HP-weighted strength, HP per species, the duel between the two active Pokémon.
- P2's team looked up among the training teams, to estimate its hidden Pokémon.
- A knock-out detector trained on P1's side, to estimate P2's unrecorded knock-outs.

Run: `python second_version/train.py` (needs `pip install catboost`)
