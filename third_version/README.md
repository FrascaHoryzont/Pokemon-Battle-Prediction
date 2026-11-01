# Third version

Result: 69.12% ± 0.69% cross-validated accuracy (5 folds × 3 shuffles) and 69.26% aggregated OOF accuracy, up from 68.64% for the second version.

Models: 15-fold bagged ensemble averaging soft probabilities of XGBoost (60%, depth 4), LightGBM (30%, 12 leaves), and CatBoost (10%, depth 5) with calibrated decision threshold (0.505).

Features (200):
- The second version's core: base battle state, survivors HP per species, duel features, and P2 hidden roster lookup.
- Soft P2 reconstruction & enhanced KO detector: expanded 15k-team unsupervised library with continuous Bayesian co-occurrence matrix and self-destruct/Spikes-aware faint detection.
- Generation 2 OU domain mechanics:
  - Snorlax factor: presence, remaining HP, and living counters alive on each side.
  - Sleep mechanics: distinguishes healthy Rest recovery (100% HP) from hostile sleep moves and Sleep Clause.
  - Advanced Spikes: Flying-type immunities, Ghost spinblockers, and bench Pokémon "Dead on Entry" (HP ≤ 12.5%).
  - True Gen 2 Physical/Special split: type-based bulk and offensive potential with 50% Attack cut for burned physical attackers.
  - Win condition extinction: uncontested sweepers (Snorlax, Zapdos/Raikou, Marowak, Machamp) free of opposing checks.
  - Volatile effects & items: Substitute, Trapped, Leech Seed, Perish Trap, and P1 held items (Thick Club, status berries).
- Timeline dynamics & initiative: HP derivative (turns 21–30 vs 1–10), recent damage velocity ratio, switch pressure, and turn 30 faster-sweeper KO threat.

Removed noisy features (pruned 270 from V2's 470):
- Ultra-rare move indicators with near-zero frequency (e.g. acidarmor, sweetscent, bubblebeam, defensecurl, constrict) that caused tree split dilution.
- Non-OU and marginal species columns with zero predictive gain (e.g. Meganium, Ampharos, Aerodactyl, Typhlosion).
- Redundant move columns, condensed into 5 consolidated tactical macro-archetypes (Recovery, Phazing, Setup, Disruption, Sacrifice).

Run: `python third_version/train.py` (needs `pip install xgboost lightgbm catboost`)
