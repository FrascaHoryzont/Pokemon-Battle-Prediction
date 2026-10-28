# First version

Result: 66.88% ± 0.61% cross-validated accuracy (5 folds × 3 shuffles).

Model: logistic regression (scaled features, C = 0.1).

Features (71): fainted Pokémon and team HP left per side (with P2's hidden faints counted), statuses, HP and boosts of the active Pokémon, reserves, and average team strength.

Run: `python first_version/train.py`