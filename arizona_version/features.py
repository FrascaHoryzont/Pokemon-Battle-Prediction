"""Feature engineering: raw battle rows -> flat numeric DataFrame.
"""

import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
DAMAGING_CATEGORIES = {"physical", "special"}
TRACKED_EFFECTS = ["confusion", "leechseed", "substitute", "perish", "protect"]


def build_type_chart_lookup(type_chart_df: pd.DataFrame) -> dict:
    return {
        (row.attacking_type, row.defending_type): row.multiplier
        for row in type_chart_df.itertuples()
    }


def type_effectiveness(atk_types, def_types, chart: dict) -> float:
    # best case: attacker uses whichever of its types hits hardest
    best = 0.0
    for at in atk_types:
        if at in ("notype", "threequestionmarks"):
            continue
        mult = 1.0
        for dt in def_types:
            if dt in ("notype", "threequestionmarks"):
                continue
            mult *= chart.get((at, dt), 1.0)
        best = max(best, mult)
    return best


# Gen 1/2 stat-stage table: stage n -> multiplier
def boost_multiplier(stage: int) -> float:
    stage = max(-6, min(6, stage))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def build_species_lookup(*dfs: pd.DataFrame) -> dict:
    """species name -> types/base stats, collected from every team/lead listing."""
    lookup = {}
    for df in dfs:
        for team, lead in zip(df["p1_team_details"], df["p2_lead_details"]):
            for mon in list(team) + [lead]:
                name = mon["name"]
                if name not in lookup:
                    lookup[name] = {
                        "types": tuple(mon["types"]),
                        **{s: mon[f"base_{s}"] for s in STATS},
                        "bst": sum(mon[f"base_{s}"] for s in STATS),
                    }
    return lookup


_UNKNOWN_SPECIES = {"types": ("notype", "notype"), **{s: 0 for s in STATS}, "bst": 0}


def _lookup_species(name: str, species_lookup: dict) -> dict:
    return species_lookup.get(name, _UNKNOWN_SPECIES)


def _team_static_features(team: list[dict]) -> dict:
    feats = {}
    for s in STATS:
        vals = [mon[f"base_{s}"] for mon in team]
        feats[f"p1_{s}_sum"] = sum(vals)
        feats[f"p1_{s}_max"] = max(vals)
        feats[f"p1_{s}_min"] = min(vals)
    feats["p1_bst_sum"] = sum(sum(mon[f"base_{s}"] for s in STATS) for mon in team)

    types = {t for mon in team for t in mon["types"] if t not in ("notype", "threequestionmarks")}
    feats["p1_type_diversity"] = len(types)
    feats["p1_dual_type_count"] = sum(1 for mon in team if "notype" not in mon["types"])

    items = [mon.get("item") for mon in team]
    feats["p1_has_leftovers"] = sum(1 for i in items if i == "leftovers")
    feats["p1_item_diversity"] = len(set(items))
    return feats


def _lead_static_features(lead: dict) -> dict:
    feats = {f"p2lead_{s}": lead[f"base_{s}"] for s in STATS}
    feats["p2lead_bst"] = sum(lead[f"base_{s}"] for s in STATS)
    feats["p2lead_dual_type"] = int("notype" not in lead["types"])
    return feats


def _matchup_features(team: list[dict], lead: dict, chart: dict) -> dict:
    atk_on_lead = [type_effectiveness(mon["types"], lead["types"], chart) for mon in team]
    lead_on_mon = [type_effectiveness(lead["types"], mon["types"], chart) for mon in team]
    return {
        "p1_vs_lead_atk_mean": sum(atk_on_lead) / len(atk_on_lead),
        "p1_vs_lead_atk_max": max(atk_on_lead),
        "p1_vs_lead_atk_super_count": sum(1 for m in atk_on_lead if m >= 2.0),
        "lead_vs_p1_atk_mean": sum(lead_on_mon) / len(lead_on_mon),
        "lead_vs_p1_atk_max": max(lead_on_mon),
        "lead_vs_p1_resisted_count": sum(1 for m in lead_on_mon if m <= 0.5),
    }


def _side_timeline_features(timeline: list[dict], side: str, team_names: set, species_lookup: dict):
    # side: "p1" or "p2". team_names is the known 6-mon roster for p1; for p2 it's
    # empty since only the lead is given up front - unrevealed p2 mons are assumed
    # to still be at full HP.
    state_key, move_key = f"{side}_pokemon_state", f"{side}_move_details"

    last_hp, fainted = {}, set()
    switch_count = 0
    prev_name, prev_hp = None, None
    status_turns = 0
    nonneutral_effect_turns = 0
    hp_by_turn = []
    dmg_power, dmg_acc, stab_hits, dmg_moves = 0.0, 0.0, 0, 0
    cat_counts = {"physical": 0, "special": 0, "status": 0}
    last_boosts = {}
    max_boost_sum = 0
    faint_turns = []
    cumulative_hp_loss = 0.0  # HP lost while the same mon stays active, not switch resets
    cumulative_healing = 0.0  # HP regained the same way (Rest/Leftovers/Recover)
    expected_dmg_sum = 0.0
    effect_counts = {e: 0 for e in TRACKED_EFFECTS}
    last_name = None
    critical_turns = 0
    move_names_used = set()

    for t in timeline:
        st = t[state_key]
        name = st["name"]
        last_hp[name] = st["hp_pct"]
        hp_by_turn.append(st["hp_pct"])
        if st["hp_pct"] < 0.5:
            critical_turns += 1
        if st["status"] == "fnt":
            if name not in fainted:
                faint_turns.append(t["turn"])
            fainted.add(name)
        elif st["status"] != "nostatus":
            status_turns += 1
        if any(e != "noeffect" for e in st["effects"]):
            nonneutral_effect_turns += 1
        for e in st["effects"]:
            if e in effect_counts:
                effect_counts[e] += 1
        if prev_name is not None and name == prev_name:
            if st["hp_pct"] < prev_hp:
                cumulative_hp_loss += prev_hp - st["hp_pct"]
            elif st["hp_pct"] > prev_hp:
                cumulative_healing += st["hp_pct"] - prev_hp
        elif prev_name is not None:
            switch_count += 1
        prev_name, prev_hp = name, st["hp_pct"]
        last_boosts = st["boosts"]
        max_boost_sum = max(max_boost_sum, sum(st["boosts"].values()))
        last_name = name

        mv = t[move_key]
        if mv is not None:
            move_names_used.add(mv["name"])
            cat_counts[mv["category"]] += 1
            if mv["category"] in DAMAGING_CATEGORIES:
                dmg_power += mv["base_power"]
                dmg_acc += mv["accuracy"]
                expected_dmg_sum += mv["base_power"] * mv["accuracy"]
                dmg_moves += 1
                mon_types = _lookup_species(name, species_lookup)["types"]
                if mv["type"] in mon_types:
                    stab_hits += 1

    revealed = set(last_hp.keys())
    unrevealed = team_names - revealed if team_names else set()
    estimated_hp_sum = sum(0.0 if n in fainted else last_hp[n] for n in revealed) + len(unrevealed) * 1.0

    first10 = sum(hp_by_turn[:10]) / len(hp_by_turn[:10])
    last10 = sum(hp_by_turn[-10:]) / len(hp_by_turn[-10:])

    n_known = len(team_names) if team_names else len(revealed)
    last_species = _lookup_species(last_name, species_lookup)
    effective_speed = last_species["spe"] * boost_multiplier(last_boosts.get("spe", 0))
    feats = {
        f"{side}_n_revealed": len(revealed),
        f"{side}_n_fainted": len(fainted),
        f"{side}_n_alive_known": n_known - len(fainted),
        f"{side}_switch_count": switch_count,
        f"{side}_status_turns": status_turns,
        f"{side}_nonneutral_effect_turns": nonneutral_effect_turns,
        f"{side}_estimated_hp_sum": estimated_hp_sum,
        f"{side}_hp_first10_avg": first10,
        f"{side}_hp_last10_avg": last10,
        f"{side}_hp_trend": last10 - first10,
        f"{side}_phys_moves": cat_counts["physical"],
        f"{side}_spec_moves": cat_counts["special"],
        f"{side}_status_moves": cat_counts["status"],
        f"{side}_dmg_move_avg_power": dmg_power / dmg_moves if dmg_moves else 0.0,
        f"{side}_dmg_move_avg_acc": dmg_acc / dmg_moves if dmg_moves else 0.0,
        f"{side}_stab_rate": stab_hits / dmg_moves if dmg_moves else 0.0,
        f"{side}_last_boost_sum": sum(last_boosts.values()) if last_boosts else 0,
        f"{side}_final_hp": hp_by_turn[-1],
        f"{side}_first_faint_turn": faint_turns[0] if faint_turns else 0,
        f"{side}_last_faint_turn": faint_turns[-1] if faint_turns else 0,
        f"{side}_cumulative_hp_loss": cumulative_hp_loss,
        f"{side}_cumulative_healing": cumulative_healing,
        f"{side}_critical_turns_frac": critical_turns / len(hp_by_turn),
        f"{side}_move_diversity": len(move_names_used),
        f"{side}_expected_dmg_avg": expected_dmg_sum / dmg_moves if dmg_moves else 0.0,
        f"{side}_max_boost_sum": max_boost_sum,
        f"{side}_effective_speed": effective_speed,
        **{f"{side}_{e}_turns": c for e, c in effect_counts.items()},
    }
    return feats, last_species["types"]


def _revealed_p2_strength_features(timeline: list[dict], species_lookup: dict) -> dict:
    revealed = {t["p2_pokemon_state"]["name"] for t in timeline}
    stats = [_lookup_species(n, species_lookup) for n in revealed]
    bst_vals = [s["bst"] for s in stats]
    return {
        "p2_revealed_bst_sum": sum(bst_vals),
        "p2_revealed_bst_mean": sum(bst_vals) / len(bst_vals) if bst_vals else 0.0,
        "p2_unrevealed_count": max(0, 6 - len(revealed)),
    }


def build_features(df: pd.DataFrame, species_lookup: dict, chart: dict) -> pd.DataFrame:
    rows = []
    for row in df.itertuples():
        team, lead, timeline = row.p1_team_details, row.p2_lead_details, row.battle_timeline
        team_names = {mon["name"] for mon in team}

        p1_feats, p1_last_types = _side_timeline_features(timeline, "p1", team_names, species_lookup)
        p2_feats, p2_last_types = _side_timeline_features(timeline, "p2", set(), species_lookup)

        feats = {"battle_id": row.battle_id}
        feats.update(_team_static_features(team))
        feats.update(_lead_static_features(lead))
        feats.update(_matchup_features(team, lead, chart))
        feats.update(p1_feats)
        feats.update(p2_feats)
        feats.update(_revealed_p2_strength_features(timeline, species_lookup))
        feats["current_matchup_p1_on_p2"] = type_effectiveness(p1_last_types, p2_last_types, chart)
        feats["current_matchup_p2_on_p1"] = type_effectiveness(p2_last_types, p1_last_types, chart)
        rows.append(feats)

    out = pd.DataFrame(rows)
    out["alive_diff"] = out["p1_n_alive_known"] - (out["p2_n_revealed"] - out["p2_n_fainted"])
    out["estimated_hp_diff"] = out["p1_estimated_hp_sum"] - out["p2_estimated_hp_sum"]
    out["hp_trend_diff"] = out["p1_hp_trend"] - out["p2_hp_trend"]
    out["p1_damage_dealt"] = out["p2_cumulative_hp_loss"]
    out["p2_damage_dealt"] = out["p1_cumulative_hp_loss"]
    out["damage_dealt_diff"] = out["p1_damage_dealt"] - out["p2_damage_dealt"]
    out["first_faint_turn_diff"] = out["p1_first_faint_turn"] - out["p2_first_faint_turn"]
    out["effective_speed_diff"] = out["p1_effective_speed"] - out["p2_effective_speed"]
    out["bst_diff"] = out["p1_bst_sum"] - (out["p2_revealed_bst_sum"] + out["p2_unrevealed_count"] * (out["p2_revealed_bst_sum"] / out["p2_n_revealed"].replace(0, 1)))

    # ratios instead of raw diffs - trees split on a bounded share more cleanly
    hp_total = (out["p1_estimated_hp_sum"] + out["p2_estimated_hp_sum"]).replace(0, 1)
    out["p1_hp_share"] = out["p1_estimated_hp_sum"] / hp_total
    alive_total = (out["p1_n_alive_known"] + out["p2_n_alive_known"]).replace(0, 1)
    out["p1_alive_share"] = out["p1_n_alive_known"] / alive_total
    dmg_total = (out["p1_damage_dealt"] + out["p2_damage_dealt"]).replace(0, 1)
    out["p1_damage_share"] = out["p1_damage_dealt"] / dmg_total
    return out
