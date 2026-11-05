"""Timeline dynamics, momentum trajectory, turn 30 initiative, and tactical move archetypes."""

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import data_extr  # noqa: E402

RECOVERY_MOVES = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"}
PHAZING_MOVES = {"roar", "whirlwind", "spikes", "rapidspin"}
SETUP_MOVES = {"curse", "swordsdance", "growth", "bellydrum", "agility", "amnesia", "barrier", "acidarmor"}
DISRUPTION_MOVES = {"toxic", "thunderwave", "glare", "hypnosis", "lovelykiss", "sleeppowder", "spore", "confuseray", "stunspore"}
SACRIFICE_MOVES = {"explosion", "selfdestruct", "destinybond"}

TOP_META_MOVES = [
    "spikes", "rapidspin", "curse", "rest", "sleeptalk",
    "explosion", "thunderwave", "toxic", "roar", "whirlwind"
]


def timeline_momentum_features(battle: dict) -> dict[str, float]:
    """Momentum trajectory: early vs late HP differentials, recent damage velocity, and switch pressure."""
    timeline = battle["battle_timeline"]
    n_turns = len(timeline)
    if n_turns == 0:
        return {
            "momentum_hp_derivative": 0.0,
            "damage_ratio_recent": 1.0,
            "switches_recent_diff": 0.0,
            "recent_hp_p1_mean": 0.0,
            "recent_hp_p2_mean": 0.0
        }

    hp_diffs = []
    p1_damages = []
    p2_damages = []
    p1_switches = []
    p2_switches = []

    last_hp = {"p1": timeline[0]["p1_pokemon_state"]["hp_pct"], "p2": timeline[0]["p2_pokemon_state"]["hp_pct"]}
    last_name = {"p1": timeline[0]["p1_pokemon_state"]["name"], "p2": timeline[0]["p2_pokemon_state"]["name"]}

    for i, turn in enumerate(timeline):
        s1 = turn["p1_pokemon_state"]
        s2 = turn["p2_pokemon_state"]
        hp_diffs.append(s1["hp_pct"] - s2["hp_pct"])

        if i > 0:
            sw1 = int(s1["name"] != last_name["p1"])
            sw2 = int(s2["name"] != last_name["p2"])
            p1_switches.append(sw1)
            p2_switches.append(sw2)

            dmg_p1 = max(0.0, last_hp["p1"] - s1["hp_pct"]) if not sw1 else 0.0
            dmg_p2 = max(0.0, last_hp["p2"] - s2["hp_pct"]) if not sw2 else 0.0
            p1_damages.append(dmg_p1)
            p2_damages.append(dmg_p2)
        else:
            p1_switches.append(0)
            p2_switches.append(0)
            p1_damages.append(0.0)
            p2_damages.append(0.0)

        last_hp["p1"] = s1["hp_pct"]
        last_hp["p2"] = s2["hp_pct"]
        last_name["p1"] = s1["name"]
        last_name["p2"] = s2["name"]

    early_diff = sum(hp_diffs[:10]) / min(10, len(hp_diffs))
    late_diff = sum(hp_diffs[-10:]) / min(10, len(hp_diffs))

    # Recent damage in turns 21-30: damage dealt by P1 is damage taken by P2
    dmg_dealt_by_p1 = sum(p2_damages[-10:])
    dmg_dealt_by_p2 = sum(p1_damages[-10:])
    damage_ratio = (dmg_dealt_by_p1 + 0.05) / (dmg_dealt_by_p2 + 0.05)

    sw_p1_recent = sum(p1_switches[-10:])
    sw_p2_recent = sum(p2_switches[-10:])

    return {
        "early_hp_diff_mean": early_diff,
        "late_hp_diff_mean": late_diff,
        "momentum_hp_derivative": late_diff - early_diff,
        "damage_ratio_recent": damage_ratio,
        "damage_recent_diff": dmg_dealt_by_p1 - dmg_dealt_by_p2,
        "switches_recent_diff": float(sw_p2_recent - sw_p1_recent),
        "recent_hp_p1_mean": sum(t["p1_pokemon_state"]["hp_pct"] for t in timeline[-10:]) / min(10, n_turns),
        "recent_hp_p2_mean": sum(t["p2_pokemon_state"]["hp_pct"] for t in timeline[-10:]) / min(10, n_turns)
    }


def turn_30_initiative_features(battle: dict) -> dict[str, float]:
    """Evaluates who holds the initiative at turn 30 and immediate knockout threats for turn 31."""
    timeline = battle["battle_timeline"]
    if len(timeline) < 2:
        return {"p1_switched_turn_30": 0.0, "p2_switched_turn_30": 0.0, "duel_fast_ko_threat": 0.0}

    last = timeline[-1]
    prev = timeline[-2]

    p1_switched = 1.0 if last["p1_pokemon_state"]["name"] != prev["p1_pokemon_state"]["name"] else 0.0
    p2_switched = 1.0 if last["p2_pokemon_state"]["name"] != prev["p2_pokemon_state"]["name"] else 0.0

    # Calculate speeds of active Pokémon
    def get_spe(side, state):
        name = state["name"]
        d = (next((p for p in battle["p1_team_details"] if p["name"] == name), None) if side == "p1"
             else (battle["p2_lead_details"] if name == battle["p2_lead_details"]["name"]
                   else data_extr.training_species_table().get(name)))
        base = d["base_spe"] if d else 80
        boost = state["boosts"].get("spe", 0)
        mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
        speed = base * mult
        if state["status"] == "par":
            speed *= 0.25
        return speed

    s1 = get_spe("p1", last["p1_pokemon_state"])
    s2 = get_spe("p2", last["p2_pokemon_state"])

    # Threat check: faster Pokémon threatening super-effective lethal hit
    t1 = data_extr.types_of(battle, "p1", last["p1_pokemon_state"]["name"]) or []
    t2 = data_extr.types_of(battle, "p2", last["p2_pokemon_state"]["name"]) or []

    threat = 0.0
    if s1 > s2 and last["p2_pokemon_state"]["hp_pct"] <= 0.40:
        eff = max((data_extr.effectiveness(t, t2) for t in t1 if t != data_extr.NO_TYPE), default=1.0)
        if eff >= 2.0:
            threat = 1.0
    elif s2 > s1 and last["p1_pokemon_state"]["hp_pct"] <= 0.40:
        eff = max((data_extr.effectiveness(t, t1) for t in t2 if t != data_extr.NO_TYPE), default=1.0)
        if eff >= 2.0:
            threat = -1.0

    return {
        "p1_switched_turn_30": p1_switched,
        "p2_switched_turn_30": p2_switched,
        "initiative_diff_turn_30": p2_switched - p1_switched,
        "duel_fast_ko_threat": threat
    }


def tactical_archetype_features(battle: dict) -> dict[str, float]:
    """Aggregates all moves used into 5 tactical categories and tracks top 10 meta moves."""
    counts = {"p1": defaultdict(float), "p2": defaultdict(float)}
    top_moves = {"p1": defaultdict(float), "p2": defaultdict(float)}

    for turn in battle["battle_timeline"]:
        for side in ("p1", "p2"):
            move = turn[f"{side}_move_details"]
            if not move:
                continue
            name = move["name"]

            if name in RECOVERY_MOVES:
                counts[side]["recovery"] += 1.0
            if name in PHAZING_MOVES:
                counts[side]["phazing"] += 1.0
            if name in SETUP_MOVES:
                counts[side]["setup"] += 1.0
            if name in DISRUPTION_MOVES:
                counts[side]["disruption"] += 1.0
            if name in SACRIFICE_MOVES:
                counts[side]["sacrifice"] += 1.0

            if name in TOP_META_MOVES:
                top_moves[side][name] = 1.0

    f = {}
    categories = ["recovery", "phazing", "setup", "disruption", "sacrifice"]
    for cat in categories:
        f[f"p1_tactical_{cat}"] = counts["p1"][cat]
        f[f"p2_tactical_{cat}"] = counts["p2"][cat]
        f[f"tactical_{cat}_diff"] = counts["p1"][cat] - counts["p2"][cat]

    for m in TOP_META_MOVES:
        f[f"meta_p1_used_{m}"] = top_moves["p1"][m]
        f[f"meta_p2_used_{m}"] = top_moves["p2"][m]
        f[f"meta_{m}_diff"] = top_moves["p1"][m] - top_moves["p2"][m]

    return f
