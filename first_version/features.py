"""Features of the first version: the battle's position at turn 30 and the strength of both teams.

Built on the shared extractor (`src/data_extr.py`). `build_table(battles)`
returns one row per battle with `battle_id`, the 71 features, and
`player_won` when known. The label is added only after the features are
computed, so no feature can see it.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data_extr  # noqa: E402  (needs the path above)

TEAM_SIZE = data_extr.TEAM_SIZE
STATUSES = data_extr.STATUSES
STAT_KEYS = data_extr.STAT_KEYS
BOOST_STATS = ["atk", "def", "spa", "spd", "spe"]
STAT_NAMES = [k.removeprefix("base_") for k in STAT_KEYS]

# A Pokémon counts as "healthy" above this HP (and with no status).
HEALTHY_HP = 0.5


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _base_total(p: dict) -> int:
    return sum(p[k] for k in STAT_KEYS)


def fainted_and_hp(battle: dict) -> dict:
    """Recorded fainted count and team HP left per side, plus P1-minus-P2 differences.

    HP left is in "Pokémon worth" (6.0 = all six at full HP); Pokémon not
    seen yet count as full.
    """
    latest = data_extr.latest_states(battle)
    f = {}
    for side in ("p1", "p2"):
        seen = latest[side].values()
        f[f"state_{side}_fainted"] = sum(s["status"] == "fnt" for s in seen)
        f[f"state_{side}_hp_left"] = sum(s["hp_pct"] for s in seen) + (TEAM_SIZE - len(seen))
    f["state_fainted_diff"] = f["state_p1_fainted"] - f["state_p2_fainted"]
    f["state_hp_left_diff"] = f["state_p1_hp_left"] - f["state_p2_hp_left"]
    return f


def hidden_faint_features(battle: dict) -> dict:
    """P2's hidden faints, and P2's fainted count corrected with them (comparable with P1's)."""
    counts = data_extr.fainted_counts(battle)
    return {
        "state_p2_hidden_faints": counts["p2_hidden"],
        "state_p2_fainted_est": counts["p2"],
        "state_fainted_diff_est": counts["p1"] - counts["p2"],
    }


def status_features(battle: dict) -> dict:
    """How many of each side's living Pokémon have each status at turn 30, plus the total."""
    f = {}
    for side, states in data_extr.alive_states(battle).items():
        for status in STATUSES:
            f[f"state_{side}_n_{status}"] = sum(s["status"] == status for s in states)
        f[f"state_{side}_n_statused"] = sum(s["status"] in STATUSES for s in states)
    f["state_statused_diff"] = f["state_p1_n_statused"] - f["state_p2_n_statused"]
    return f


def active_features(battle: dict) -> dict:
    """HP and stat boosts of the two Pokémon on the field at turn 30 (boosts reset on switching)."""
    last_turn = battle["battle_timeline"][-1]
    f = {}
    for side in ("p1", "p2"):
        state = last_turn[f"{side}_pokemon_state"]
        f[f"state_{side}_active_hp"] = state["hp_pct"]
        for stat in BOOST_STATS:
            f[f"state_{side}_active_boost_{stat}"] = state["boosts"].get(stat, 0)
        f[f"state_{side}_active_boost_total"] = sum(state["boosts"].get(k, 0) for k in BOOST_STATS)
    f["state_active_hp_diff"] = f["state_p1_active_hp"] - f["state_p2_active_hp"]
    f["state_active_boost_diff"] = f["state_p1_active_boost_total"] - f["state_p2_active_boost_total"]
    return f


def reserve_features(battle: dict) -> dict:
    """Fresh Pokémon in the back and healthy Pokémon per side.

    P1's unused Pokémon (count and summed base stats), P2's unrevealed count,
    and "healthy" Pokémon: alive, above HEALTHY_HP, no status (unseen ones included).
    """
    seen = data_extr.latest_states(battle)
    unused = [p for p in battle["p1_team_details"] if p["name"] not in seen["p1"]]
    f = {
        "state_p1_unused": len(unused),
        "state_p1_unused_base_total": sum(_base_total(p) for p in unused),
        "state_p2_unrevealed": TEAM_SIZE - len(seen["p2"]),
    }
    for side, states in data_extr.alive_states(battle).items():
        healthy = sum(s["hp_pct"] > HEALTHY_HP and s["status"] == "nostatus" for s in states)
        f[f"state_{side}_healthy"] = healthy + TEAM_SIZE - len(seen[side])
    f["state_healthy_diff"] = f["state_p1_healthy"] - f["state_p2_healthy"]
    return f


def strength_features(battle: dict) -> dict:
    """Average base stats of P1's team and P2's revealed Pokémon, and of each side's survivors."""
    p1 = battle["p1_team_details"]
    p2, missing = data_extr.p2_revealed(battle)
    alive = {side: {s["name"] for s in states} for side, states in data_extr.alive_states(battle).items()}
    # P1's unused Pokémon are alive too, but have no recorded state yet.
    alive["p1"] |= {p["name"] for p in p1 if p["name"] not in data_extr.latest_states(battle)["p1"]}

    f = {"team_p2_unknown_species": missing}
    for stat, key in zip(STAT_NAMES, STAT_KEYS):
        f[f"team_p1_mean_{stat}"] = _mean(p[key] for p in p1)
        f[f"team_p2_mean_{stat}"] = _mean(p[key] for p in p2)
        f[f"team_mean_{stat}_diff"] = f[f"team_p1_mean_{stat}"] - f[f"team_p2_mean_{stat}"]
    f["team_p1_mean_total"] = _mean(_base_total(p) for p in p1)
    f["team_p2_mean_total"] = _mean(_base_total(p) for p in p2)
    f["team_mean_total_diff"] = f["team_p1_mean_total"] - f["team_p2_mean_total"]
    f["team_p1_alive_mean_total"] = _mean(_base_total(p) for p in p1 if p["name"] in alive["p1"])
    f["team_p2_alive_mean_total"] = _mean(_base_total(p) for p in p2 if p["name"] in alive["p2"])
    f["team_alive_mean_total_diff"] = f["team_p1_alive_mean_total"] - f["team_p2_alive_mean_total"]
    return f


FEATURE_FUNCTIONS = [fainted_and_hp, hidden_faint_features, status_features, active_features,
                     reserve_features, strength_features]


def build_table(battles: list[dict]) -> pd.DataFrame:
    """One row per battle: `battle_id`, every feature, and `player_won` if the battle has it."""
    rows = []
    for battle in battles:
        row = {"battle_id": battle["battle_id"]}
        for func in FEATURE_FUNCTIONS:
            row.update(func(battle))
        if "player_won" in battle:
            row["player_won"] = int(battle["player_won"])
        rows.append(row)
    return pd.DataFrame(rows)
