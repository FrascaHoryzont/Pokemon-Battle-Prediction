"""Master feature pipeline for Version 3.

Combines:
1. Base state (fainted counts, HP left, status, active boosts, reserves, strength)
2. Gen 2 OU domain features:
   - Snorlax factor and counter availability
   - Rest recovery vs hostile sleep & Sleep Clause
   - Advanced Spikes (flying immunity, Ghost spinblockers, Dead on Entry)
   - True physical/special split by Gen 2 type & Burn reduction
   - Win condition extinction: "Sweeper alive while opposing checks are eliminated"
   - Fatal defensive type gaps (zero resists to core attack types)
3. Soft P2 reconstruction & enhanced KO detector:
   - Unsupervised 15,000 team library (train + test P1 teams)
   - Co-occurrence matrix P(species_B | species_A)
   - Soft Jaccard fallback
   - Enhanced KO detector with Spikes & self-destruct awareness
4. Timeline dynamics & initiative:
   - Momentum derivative (turns 21-30 vs 1-10), recent damage ratio, switch pressure
   - Turn 30 initiative & immediate KO threat
   - Tactical archetypes (recovery, phazing, setup, disruption, sacrifice)
5. Granular survivor & duel features:
   - species_hp: exact remaining HP of each surviving species
   - p2_full_team: probability of each species being in P2's roster
   - duel_features & survivor_features: matchups, speeds, and offensive/defensive bulk
   - move indicators appearing in >= 100 training battles
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from first_version.features import build_table as first_version_table  # noqa: E402
from second_version.features import (  # noqa: E402
    _indicator_tables,
    duel_features,
    p2_full_team,
    species_hp,
    survivor_features,
)
from fourth_version.features_domain import (  # noqa: E402
    advanced_spikes_features,
    defensive_gap_features,
    physical_special_split_features,
    rest_and_sleep_features,
    snorlax_features,
    volatile_effects_and_items_features,
)
from fourth_version.features_lookup import (  # noqa: E402
    build_team_library,
    enhanced_detector_features,
    soft_lookup_features,
)
from fourth_version.features_timeline import (  # noqa: E402
    tactical_archetype_features,
    timeline_momentum_features,
    turn_30_initiative_features,
)


def _move_indicators(battle: dict) -> dict:
    """1 for each move used per side."""
    found = {}
    for turn in battle["battle_timeline"]:
        for side in ("p1", "p2"):
            move = turn[f"{side}_move_details"]
            if move:
                found[f"bag_{side}_used_{move['name']}"] = 1
    return found


def extract_batch_domain_features(battles: list[dict], library: dict) -> pd.DataFrame:
    """Extracts all domain, soft lookup, and timeline features for a batch of battles."""
    rows = []
    for b in battles:
        row = {}
        # Domain features & Win Conditions
        row.update(snorlax_features(b))
        row.update(rest_and_sleep_features(b))
        row.update(advanced_spikes_features(b))
        row.update(physical_special_split_features(b))
        row.update(defensive_gap_features(b))
        row.update(volatile_effects_and_items_features(b))

        # Soft lookup & enhanced KO detector
        row.update(soft_lookup_features(b, library))
        row.update(enhanced_detector_features(b))

        # Timeline momentum & initiative
        row.update(timeline_momentum_features(b))
        row.update(turn_30_initiative_features(b))
        row.update(tactical_archetype_features(b))

        # Matchups and duel
        row.update(survivor_features(b))
        row.update(duel_features(b))

        rows.append(row)

    return pd.DataFrame(rows)


def build_tables(train_battles: list[dict], test_battles: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates the unified, high-accuracy V3 feature matrices directly from raw battle data."""
    print("Building team library strictly from training battles...")
    library = build_team_library(train_battles)

    print("Computing first-version base features...")
    train_first = first_version_table(train_battles)
    test_first = first_version_table(test_battles)

    print("Extracting domain, soft lookup, timeline and duel features...")
    train_domain = extract_batch_domain_features(train_battles, library)
    test_domain = extract_batch_domain_features(test_battles, library)

    print("Extracting granular species HP, P2 full team, and move indicators...")
    full_train, full_test = _indicator_tables(train_battles, test_battles, p2_full_team)
    hp_train, hp_test = _indicator_tables(train_battles, test_battles, species_hp)
    moves_train, moves_test = _indicator_tables(train_battles, test_battles, _move_indicators)

    # Separate labels and IDs
    train_id = train_first["battle_id"]
    test_id = test_first["battle_id"]
    train_y = train_first["player_won"] if "player_won" in train_first.columns else None

    # Merge all feature tables
    X_train = pd.concat([
        train_first.drop(columns=["battle_id", "player_won"], errors="ignore"),
        train_domain,
        full_train,
        hp_train,
        moves_train
    ], axis=1)

    X_test = pd.concat([
        test_first.drop(columns=["battle_id", "player_won"], errors="ignore"),
        test_domain,
        full_test,
        hp_test,
        moves_test
    ], axis=1)

    # Fill NaNs/Infs
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Prune strictly constant columns on train (only if substantial dataset)
    if len(X_train) > 100:
        constant_cols = {c for c in X_train.columns if X_train[c].min() == X_train[c].max()}
        keep_cols = sorted(c for c in X_train.columns if c not in constant_cols)
    else:
        keep_cols = sorted(X_train.columns.tolist())

    X_train = X_train[keep_cols]
    X_test = X_test.reindex(columns=keep_cols, fill_value=0.0)

    # Assemble final outputs with IDs and labels
    train_out = pd.concat([pd.Series(train_id, name="battle_id"), X_train], axis=1)
    if train_y is not None:
        train_out["player_won"] = train_y.astype(int)

    test_out = pd.concat([pd.Series(test_id, name="battle_id"), X_test], axis=1)

    print(f"Features assembled: {len(keep_cols)} rich features (V2 granular + V3 domain)")

    return train_out, test_out
