# arizona_version

Result: ~0.673 grouped 5-fold CV accuracy.

Model: LightGBM + logistic regression (equal-weight probability blend), both tuned with Optuna.

Features (~120): 
- team/lead base stats
- type-chart matchups 
- timeline attrition (faints, HP trend, damage dealt/healing, critical-HP turns, move diversity)
- a species stats lookup to track Pokémon the opponent reveals mid-battle
- effective speed with stat boosts 
- current matchup at turn 30
- HP/alive/damage share ratios
- few volatile statuses (confusion/leech seed/substitute/perish/protect).

Run: `python arizona_version/train.py` (from repo root, needs `data/train.jsonl`, `data/test.jsonl`, `type_chart_gen2.csv`, `sample_submission.csv` in place as usual).

## Things worth checking in the shared pipeline

- **CV split**: `src/cv_splitter.py` uses plain `StratifiedKFold`. Team rosters repeat a lot in the data (one composition shows up ~195 times across 10k rows) and 41% of test rosters already appear in train — a plain split can leak a team's win rate across train/val and inflate the CV number. `validation.py` here groups by team composition instead.
- **`features_static.py`**: `p1_lead = p1_team[0]` assumes the first Pokémon listed is the one that actually leads. Checked against `battle_timeline[0]`: only true 71% of the time. The lead-vs-lead features are built on the wrong Pokémon for the other 29% of rows.
- **`data_extr.py`'s hidden-faint heuristic** (HP ≤ 0.2 + switch + never seen again ⇒ assume fainted): tested the same pattern on P1 (where the full roster is known) — the same-looking switch comes back into the battle about 51% of the time. A lot of these are just ordinary low-HP retreats, not faints, and the 30-turn cutoff adds its own bias (a Pokémon could return on turn 35, outside the visible window). Worth treating this feature with some caution.
