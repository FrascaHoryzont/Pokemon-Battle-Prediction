"""Domain features based on Generation 2 OverUsed (OU) competitive mechanics:
- Snorlax presence and health differential
- Rest recovery vs hostile sleep & sleep clause
- Spikes hazards, grounded vulnerability, Ghost spinblockers, and stealth dead on entry
- True physical/special split by Gen 2 type and burn attack reduction
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import data_extr  # noqa: E402

HOSTILE_SLEEP_MOVES = {"lovelykiss", "spore", "hypnosis", "sleeppowder", "sing"}


def _is_flying(battle: dict, side: str, name: str) -> bool:
    types = data_extr.types_of(battle, side, name) or []
    return "flying" in types


def _is_ghost(battle: dict, side: str, name: str) -> bool:
    types = data_extr.types_of(battle, side, name) or []
    return "ghost" in types


def snorlax_features(battle: dict) -> dict[str, float]:
    """Snorlax presence, HP, and diff on each team."""
    p1_team = battle["p1_team_details"]
    latest = data_extr.latest_states(battle)

    # P1 Snorlax
    p1_has = 1.0 if any(p["name"] == "snorlax" for p in p1_team) else 0.0
    p1_hp = 0.0
    if p1_has:
        if "snorlax" in latest["p1"]:
            p1_hp = latest["p1"]["snorlax"]["hp_pct"] if latest["p1"]["snorlax"]["status"] != "fnt" else 0.0
        else:
            p1_hp = 1.0  # alive and unrevealed in timeline

    # P2 Snorlax
    p2_lead_is_snorlax = (battle["p2_lead_details"]["name"] == "snorlax")
    p2_seen_snorlax = "snorlax" in latest["p2"] or p2_lead_is_snorlax
    p2_hp = 0.0
    p2_has = 1.0 if p2_seen_snorlax else 0.0
    if p2_seen_snorlax:
        if "snorlax" in latest["p2"]:
            p2_hp = latest["p2"]["snorlax"]["hp_pct"] if latest["p2"]["snorlax"]["status"] != "fnt" else 0.0
        else:
            p2_hp = 1.0

    return {
        "p1_has_snorlax": p1_has,
        "p2_has_snorlax": p2_has,
        "snorlax_hp_p1": p1_hp,
        "snorlax_hp_p2": p2_hp,
        "snorlax_hp_diff": p1_hp - p2_hp,
    }


def rest_and_sleep_features(battle: dict) -> dict[str, float]:
    """Distinguishes healthy Rest recovery from debilitating hostile sleep."""
    timeline = battle["battle_timeline"]
    alive = data_extr.alive_states(battle)

    # Track which Pokémon used Rest during the timeline
    rested_pokemon = {"p1": set(), "p2": set()}
    hostile_slept = {"p1": set(), "p2": set()}

    for turn in timeline:
        for side, other in (("p1", "p2"), ("p2", "p1")):
            own_move = turn[f"{side}_move_details"]
            opp_move = turn[f"{other}_move_details"]
            own_state = turn[f"{side}_pokemon_state"]

            if own_move and own_move["name"] == "rest":
                rested_pokemon[side].add(own_state["name"])

            if opp_move and opp_move["name"] in HOSTILE_SLEEP_MOVES:
                if own_state["status"] == "slp":
                    hostile_slept[side].add(own_state["name"])

    f = {}
    for side in ("p1", "p2"):
        slp_rest = 0.0
        slp_hostile = 0.0
        for s in alive[side]:
            if s["status"] == "slp":
                name = s["name"]
                if name in rested_pokemon[side] and s["hp_pct"] >= 0.70:
                    slp_rest += 1.0
                else:
                    slp_hostile += 1.0

        f[f"{side}_slp_from_rest"] = slp_rest
        f[f"{side}_slp_hostile"] = slp_hostile

    f["slp_hostile_diff"] = f["p2_slp_hostile"] - f["p1_slp_hostile"]
    f["slp_rest_diff"] = f["p1_slp_from_rest"] - f["p2_slp_from_rest"]
    f["p2_sleep_clause_active"] = 1.0 if f["p2_slp_hostile"] >= 1.0 else 0.0
    return f


def advanced_spikes_features(battle: dict) -> dict[str, float]:
    """Spikes active, grounded susceptibility, spinblockers, and dead-on-entry."""
    timeline = battle["battle_timeline"]
    up = {"p1": 0.0, "p2": 0.0}
    for turn in timeline:
        for side, other in (("p1", "p2"), ("p2", "p1")):
            move = turn[f"{side}_move_details"]
            if move and move["name"] == "spikes":
                up[side] = 1.0
            elif move and move["name"] == "rapidspin":
                up[other] = 0.0

    alive = data_extr.alive_states(battle)
    latest = data_extr.latest_states(battle)
    active_now = {side: timeline[-1][f"{side}_pokemon_state"]["name"] for side in ("p1", "p2")} if timeline else {}

    f = {
        "p1_spikes_active": up["p1"],
        "p2_spikes_active": up["p2"],
        "spikes_active_diff": up["p1"] - up["p2"],
    }

    # Grounded vs flying survivors
    for side in ("p1", "p2"):
        survivors = alive[side]
        total = len(survivors)
        grounded = sum(1.0 for s in survivors if not _is_flying(battle, side, s["name"]))
        f[f"{side}_spikes_grounded_ratio"] = (grounded / total) if total > 0 else 0.0

        # Spinblocker: has Spikes up and has a living Ghost Pokémon
        has_ghost = any(_is_ghost(battle, side, s["name"]) for s in survivors)
        f[f"{side}_spinblocker_active"] = 1.0 if (up[side] > 0 and has_ghost) else 0.0

    # Stealth Dead on Entry: bench Pokémon, grounded, HP <= 12.5%, under opposing Spikes
    for side, other in (("p1", "p2"), ("p2", "p1")):
        stealth_dead = 0.0
        if up[other] > 0:  # other laid spikes on side's field
            for s in alive[side]:
                if s["name"] != active_now.get(side):
                    if not _is_flying(battle, side, s["name"]) and s["hp_pct"] <= 0.125:
                        stealth_dead += 1.0
        f[f"{side}_stealth_dead_count"] = stealth_dead

    f["stealth_dead_diff"] = f["p2_stealth_dead_count"] - f["p1_stealth_dead_count"]
    return f


def physical_special_split_features(battle: dict) -> dict[str, float]:
    """Accurate physical/special bulk and offense split according to Gen 2 mechanics."""
    alive = data_extr.alive_states(battle)
    p1_team = battle["p1_team_details"]
    latest = data_extr.latest_states(battle)

    # Details lookup helper
    def get_details(side, name):
        if side == "p1":
            return next((p for p in p1_team if p["name"] == name), None)
        if name == battle["p2_lead_details"]["name"]:
            return battle["p2_lead_details"]
        return data_extr.training_species_table().get(name)

    stats = {
        "p1": {"phys_off": 0.0, "spec_off": 0.0, "phys_bulk": 0.0, "spec_bulk": 0.0},
        "p2": {"phys_off": 0.0, "spec_off": 0.0, "phys_bulk": 0.0, "spec_bulk": 0.0},
    }

    for side in ("p1", "p2"):
        # Living seen
        for s in alive[side]:
            d = get_details(side, s["name"])
            if not d:
                continue
            hp = s["hp_pct"]
            # Burn halves physical attack in Gen 2
            atk_mult = 0.5 if s["status"] == "brn" else 1.0

            stats[side]["phys_off"] += hp * d["base_atk"] * atk_mult
            stats[side]["spec_off"] += hp * d["base_spa"]
            stats[side]["phys_bulk"] += hp * (d["base_hp"] + d["base_def"])
            stats[side]["spec_bulk"] += hp * (d["base_hp"] + d["base_spd"])

        # P1 unused
        if side == "p1":
            for p in p1_team:
                if p["name"] not in latest["p1"]:
                    stats["p1"]["phys_off"] += p["base_atk"]
                    stats["p1"]["spec_off"] += p["base_spa"]
                    stats["p1"]["phys_bulk"] += (p["base_hp"] + p["base_def"])
                    stats["p1"]["spec_bulk"] += (p["base_hp"] + p["base_spd"])

    return {
        "fuel_phys_offense_diff": stats["p1"]["phys_off"] - stats["p2"]["phys_off"],
        "fuel_spec_offense_diff": stats["p1"]["spec_off"] - stats["p2"]["spec_off"],
        "fuel_phys_bulk_diff": stats["p1"]["phys_bulk"] - stats["p2"]["phys_bulk"],
        "fuel_spec_bulk_diff": stats["p1"]["spec_bulk"] - stats["p2"]["spec_bulk"],
    }


PRIMARY_ATTACKING_TYPES = ["electric", "ground", "ice", "water", "fire", "psychic", "normal", "fighting"]


def defensive_gap_features(battle: dict) -> dict[str, float]:
    """Identifies fatal defensive type gaps (zero resists or immunities to core Gen 2 attack types)."""
    alive = data_extr.alive_states(battle)
    latest = data_extr.latest_states(battle)
    p1_team = battle["p1_team_details"]

    survivor_types = {
        "p1": [data_extr.types_of(battle, "p1", s["name"]) for s in alive["p1"]] +
              [p["types"] for p in p1_team if p["name"] not in latest["p1"]],
        "p2": [data_extr.types_of(battle, "p2", s["name"]) for s in alive["p2"]]
    }

    gaps = {"p1": 0.0, "p2": 0.0}
    f = {}
    for atk_type in PRIMARY_ATTACKING_TYPES:
        for side in ("p1", "p2"):
            types_list = [t for t in survivor_types[side] if t is not None]
            has_resist = any(data_extr.effectiveness(atk_type, def_t) <= 0.5 for def_t in types_list) if types_list else True
            gap_val = 0.0 if has_resist else 1.0
            gaps[side] += gap_val
            f[f"gap_{side}_{atk_type}"] = gap_val

    f["gaps_total_p1"] = gaps["p1"]
    f["gaps_total_p2"] = gaps["p2"]
    f["gaps_diff"] = gaps["p2"] - gaps["p1"]
    return f


TRACKED_EFFECTS = ["substitute", "trapped", "confusion", "leechseed", "perish", "protect"]


def volatile_effects_and_items_features(battle: dict) -> dict[str, float]:
    """Extracts previously unmined volatile effects (substitute, trapped, leechseed, etc.) and P1 items."""
    timeline = battle["battle_timeline"]
    p1_team = battle["p1_team_details"]
    f = {}

    # Turn 30 active effects
    last_turn = timeline[-1] if timeline else None
    for eff in TRACKED_EFFECTS:
        p1_has_eff = 1.0 if (last_turn and eff in last_turn["p1_pokemon_state"].get("effects", [])) else 0.0
        p2_has_eff = 1.0 if (last_turn and eff in last_turn["p2_pokemon_state"].get("effects", [])) else 0.0
        f[f"t30_p1_eff_{eff}"] = p1_has_eff
        f[f"t30_p2_eff_{eff}"] = p2_has_eff
        f[f"t30_eff_{eff}_diff"] = p1_has_eff - p2_has_eff

    # Accumulate turns under key volatile effects across all turns
    turns_eff = {"p1": {e: 0.0 for e in TRACKED_EFFECTS}, "p2": {e: 0.0 for e in TRACKED_EFFECTS}}
    for turn in timeline:
        for side in ("p1", "p2"):
            for eff in turn[f"{side}_pokemon_state"].get("effects", []):
                if eff in turns_eff[side]:
                    turns_eff[side][eff] += 1.0

    for eff in ["substitute", "trapped", "leechseed"]:
        f[f"turns_p1_eff_{eff}"] = turns_eff["p1"][eff]
        f[f"turns_p2_eff_{eff}"] = turns_eff["p2"][eff]
        f[f"turns_eff_{eff}_diff"] = turns_eff["p1"][eff] - turns_eff["p2"][eff]

    # Perish trap combo: trapped + perish active on same Pokemon at turn 30
    p1_ptrap = 1.0 if (f["t30_p1_eff_trapped"] and f["t30_p1_eff_perish"]) else 0.0
    p2_ptrap = 1.0 if (f["t30_p2_eff_trapped"] and f["t30_p2_eff_perish"]) else 0.0
    f["t30_perish_trap_p1_suffering"] = p1_ptrap
    f["t30_perish_trap_p2_suffering"] = p2_ptrap
    f["t30_perish_trap_diff"] = p2_ptrap - p1_ptrap

    # P1 Held Items
    p1_thickclub = sum(1.0 for p in p1_team if p.get("item") == "thickclub")
    p1_berries = sum(1.0 for p in p1_team if p.get("item") in ("miracleberry", "mintberry", "przcureberry"))
    p1_lefties = sum(1.0 for p in p1_team if p.get("item") == "leftovers")

    f["p1_item_thickclub"] = p1_thickclub
    f["p1_item_status_berries"] = p1_berries
    f["p1_item_leftovers_count"] = p1_lefties

    return f
