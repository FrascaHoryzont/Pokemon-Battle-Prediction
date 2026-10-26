from data_parse import get_type_multiplier
import numpy as np
import pandas as pd

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]

# Extract static features from battles, including team stats, lead differences, and type advantages.
def extract_static_features(battles, type_chart):
  rows = []
  for b in battles:
    b_id = b["battle_id"]
    p1_team = b.get("p1_team_details", [])
    p2_lead = b.get("p2_lead_details", {})
    p1_lead = p1_team[0] if p1_team else {}

    feat = {"battle_id": b_id}

    for s in STATS:
      vals = [p.get(s, 0) for p in p1_team]
      feat[f"p1_team_mean_{s}"] = np.mean(vals)
      feat[f"p1_team_max_{s}"] = np.max(vals)
      feat[f"p1_team_std_{s}"] = np.std(vals)

    for s in STATS:
      feat[f"lead_diff_{s}"] = p1_lead.get(s, 0) - p2_lead.get(s, 0)
    feat["lead_p1_faster"] = int(
        p1_lead.get("base_spe", 0) > p2_lead.get("base_spe", 0)
    )

    feat["p1_leftovers_count"] = sum(
        1 for p in p1_team if p.get("item") == "leftovers"
    )

    p1_lead_types = p1_lead.get("types", [])
    p2_lead_types = p2_lead.get("types", [])

    p1_vs_p2_mult = max([
        get_type_multiplier(t, p2_lead_types, type_chart) for t in p1_lead_types
    ] or [1.0])
    p2_vs_p1_mult = max([
        get_type_multiplier(t, p1_lead_types, type_chart) for t in p2_lead_types
    ] or [1.0])

    feat["lead_p1_type_adv"] = p1_vs_p2_mult
    feat["lead_p2_type_adv"] = p2_vs_p1_mult
    feat["lead_type_net_adv"] = p1_vs_p2_mult - p2_vs_p1_mult

    feat["p1_team_se_vs_p2_lead"] = sum(
        1
        for p in p1_team
        if max(
            [
                get_type_multiplier(t, p2_lead_types, type_chart)
                for t in p.get("types", [])
            ]
            or [1.0]
        )
        >= 2.0
    )

    rows.append(feat)

  return pd.DataFrame(rows).set_index("battle_id")