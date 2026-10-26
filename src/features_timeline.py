import numpy as np
import pandas as pd

# Extract features from the battle timeline, including move usage, base power, status moves, and HP trends.
def extract_timeline_features(battles):
  rows = []
  for b in battles:
    b_id = b["battle_id"]
    timeline = b.get("battle_timeline", [])

    p1_null_moves = 0
    p2_null_moves = 0
    p1_base_powers = []
    p2_base_powers = []
    p1_status_moves = 0
    p2_status_moves = 0
    p1_rest_count = 0
    p2_rest_count = 0
    p1_boost_sum = 0
    p2_boost_sum = 0
    hp_diffs_by_turn = []

    for t in timeline:
      p1_state = t.get("p1_pokemon_state", {})
      p2_state = t.get("p2_pokemon_state", {})
      p1_move = t.get("p1_move_details")
      p2_move = t.get("p2_move_details")

      hp_diffs_by_turn.append(
          p1_state.get("hp_pct", 0.0) - p2_state.get("hp_pct", 0.0)
      )

      p1_boost_sum += sum(p1_state.get("boosts", {}).values())
      p2_boost_sum += sum(p2_state.get("boosts", {}).values())

      if p1_move is None:
        p1_null_moves += 1
      else:
        bp = p1_move.get("base_power", 0)
        p1_base_powers.append(bp)
        if p1_move.get("category") == "status":
          p1_status_moves += 1
        if p1_move.get("name") == "rest":
          p1_rest_count += 1

      if p2_move is None:
        p2_null_moves += 1
      else:
        bp = p2_move.get("base_power", 0)
        p2_base_powers.append(bp)
        if p2_move.get("category") == "status":
          p2_status_moves += 1
        if p2_move.get("name") == "rest":
          p2_rest_count += 1

    early_hp_diff = (
        np.mean(hp_diffs_by_turn[:10]) if len(hp_diffs_by_turn) >= 10 else 0.0
    )
    late_hp_diff = (
        np.mean(hp_diffs_by_turn[-10:]) if len(hp_diffs_by_turn) >= 10 else 0.0
    )

    rows.append({
        "battle_id": b_id,
        "p1_switches_or_null": p1_null_moves,
        "p2_switches_or_null": p2_null_moves,
        "null_moves_diff": p2_null_moves - p1_null_moves,
        "p1_mean_base_power": (
            np.mean(p1_base_powers) if p1_base_powers else 0.0
        ),
        "p2_mean_base_power": (
            np.mean(p2_base_powers) if p2_base_powers else 0.0
        ),
        "p1_status_moves": p1_status_moves,
        "p2_status_moves": p2_status_moves,
        "p1_rest_used": p1_rest_count,
        "p2_rest_used": p2_rest_count,
        "boost_sum_diff": p1_boost_sum - p2_boost_sum,
        "early_hp_diff_mean": early_hp_diff,
        "late_hp_diff_mean": late_hp_diff,
        "momentum_hp_trend": late_hp_diff - early_hp_diff,
    })

  return pd.DataFrame(rows).set_index("battle_id")