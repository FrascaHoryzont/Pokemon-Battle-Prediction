import pandas as pd

# Extract features from the end state of battles, including total HP, fainted counts, and status conditions.
def extract_endstate_features(battles):
  rows = []
  for b in battles:
    b_id = b["battle_id"]
    timeline = b.get("battle_timeline", [])

    p1_last_hp = {p["name"]: 1.0 for p in b.get("p1_team_details", [])}
    p1_last_status = {
        p["name"]: "nostatus" for p in b.get("p1_team_details", [])
    }

    p2_last_hp = {}
    p2_last_status = {}

    for turn_data in timeline:
      p1_state = turn_data.get("p1_pokemon_state", {})
      p2_state = turn_data.get("p2_pokemon_state", {})

      if p1_state and "name" in p1_state:
        p1_last_hp[p1_state["name"]] = p1_state.get("hp_pct", 0.0)
        p1_last_status[p1_state["name"]] = p1_state.get("status", "nostatus")

      if p2_state and "name" in p2_state:
        p2_last_hp[p2_state["name"]] = p2_state.get("hp_pct", 0.0)
        p2_last_status[p2_state["name"]] = p2_state.get("status", "nostatus")

    p2_revealed_count = len(p2_last_hp)
    p2_unrevealed_count = 6 - p2_revealed_count

    p1_total_hp = sum(p1_last_hp.values())
    p2_total_hp = sum(p2_last_hp.values()) + (p2_unrevealed_count * 1.0)

    p1_fainted = sum(
        1
        for name in p1_last_hp
        if p1_last_hp[name] == 0.0 or p1_last_status[name] == "fnt"
    )
    p2_fainted = sum(
        1
        for name in p2_last_hp
        if p2_last_hp[name] == 0.0 or p2_last_status[name] == "fnt"
    )

    p1_statused = sum(
        1 for s in p1_last_status.values() if s not in ["nostatus", "fnt"]
    )
    p2_statused = sum(
        1 for s in p2_last_status.values() if s not in ["nostatus", "fnt"]
    )

    last_turn = timeline[-1] if timeline else {}
    p1_end_hp = last_turn.get("p1_pokemon_state", {}).get("hp_pct", 0.0)
    p2_end_hp = last_turn.get("p2_pokemon_state", {}).get("hp_pct", 0.0)

    rows.append({
        "battle_id": b_id,
        "p1_total_hp_est": p1_total_hp,
        "p2_total_hp_est": p2_total_hp,
        "hp_total_diff": p1_total_hp - p2_total_hp,
        "p1_fainted_count": p1_fainted,
        "p2_fainted_count": p2_fainted,
        "fainted_diff": p2_fainted - p1_fainted,
        "p2_revealed_pokemon": p2_revealed_count,
        "p1_statused_alive": p1_statused,
        "p2_statused_alive": p2_statused,
        "end_turn_active_hp_diff": p1_end_hp - p2_end_hp,
    })

  return pd.DataFrame(rows).set_index("battle_id")