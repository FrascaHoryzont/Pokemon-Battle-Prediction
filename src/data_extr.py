"""
Shared extractor: gives every feature module the same basic facts about a battle.

How to use it: load the battles once, then call the functions you need on
each battle `b` inside your feature loop.

    from data_extr import load_battles, fainted_counts, turn_table

    battles = load_battles("train")      # or "test"
    for b in battles:
        fainted = fainted_counts(b)      # {"p1": 2, "p2": 1, ...}
        turns = turn_table(b)            # list of 30 dicts, one per turn

What each function outputs:
- load_battles(split)        -> list of all battles (finds data/ by itself, checks there are 10,000 / 5,000)
- effectiveness(type, types) -> type multiplier, e.g. effectiveness("water", ["fire", "notype"]) = 2.0
- training_species_table()   -> {species: {"types": [...], "base_hp": ..., ...}} for every known species
- latest_states(b)           -> {"p1": {name: last state}, "p2": {...}} for every Pokémon seen
- fainted_counts(b)          -> {"p1", "p2_recorded", "p2_hidden", "p2"}: fainted Pokémon at turn 30
- alive_states(b)            -> {"p1": [states], "p2": [states]} of the Pokémon not fainted
- p2_revealed(b)             -> (P2's revealed Pokémon with types and stats, number with unknown species)
- types_of(b, side, name)    -> types of one Pokémon, or None if unknown
- turn_table(b)              -> per turn and side: team HP left, damage taken, HP healed, KOs, switched

Important: P2's faints are often not recorded (the data jumps straight to the
replacement). fainted_counts(b)["p2"] and turn_table include these hidden faints;
use them instead of counting "fnt" yourself.
"""

from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TYPE_CHART_PATH = REPO_ROOT / "type_chart_gen2.csv"

TEAM_SIZE = 6
SIDES = ("p1", "p2")
OTHER_SIDE = {"p1": "p2", "p2": "p1"}
STAT_KEYS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
STATUSES = ["slp", "par", "frz", "brn", "psn", "tox"]

# Placeholder used in the data for the missing second type of single-type Pokémon.
NO_TYPE = "notype"

# A P2 Pokémon that leaves the field at or below this HP, without a recorded
# faint, and never comes back is counted as a hidden faint. 0.2 scored best in
# cross-validation among 0.1-1.0.
HIDDEN_FAINT_HP = 0.2

EXPECTED_COUNTS = {"train": 10_000, "test": 5_000}


# ---------------------------------------------------------------- data

def find_data_dir() -> Path:
    """Folder with train.jsonl / test.jsonl.

    Order: the DATA_DIR environment variable, then the repo's `data/` folder,
    then the folder that contains the repo.
    """
    if "DATA_DIR" in os.environ:
        return Path(os.environ["DATA_DIR"])
    candidates = [REPO_ROOT / "data", REPO_ROOT.parent]
    return next((d for d in candidates if (d / "train.jsonl").exists()), candidates[0])


def load_battles(split: str, data_dir: Path | None = None) -> list[dict]:
    """Return all battles of `split` ("train" or "test") as a list of dicts.

    Raises if the file is missing or the number of battles is not the expected
    one, so a wrong folder fails loudly instead of silently.
    """
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = Path(data_dir or find_data_dir()) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the data in data/ or set DATA_DIR.")
    with open(path, encoding="utf-8") as f:
        battles = [json.loads(line) for line in f if line.strip()]
    if len(battles) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} battles, got {len(battles)}")
    return battles


# ---------------------------------------------------------------- types and species

@lru_cache(maxsize=1)
def load_type_chart() -> dict:
    """Return {(attacking_type, defending_type): multiplier}; pairs not listed are 1x."""
    with open(TYPE_CHART_PATH, encoding="utf-8") as f:
        return {(r["attacking_type"], r["defending_type"]): float(r["multiplier"]) for r in csv.DictReader(f)}


def effectiveness(attack_type: str, defender_types) -> float:
    """Damage multiplier of a move of `attack_type` against a Pokémon with `defender_types`.

    Dual types multiply (Ice vs Ground/Flying = 4x); "notype" is ignored.
    Types missing from the chart, like Curse's "threequestionmarks", are neutral.
    """
    chart = load_type_chart()
    multiplier = 1.0
    for t in defender_types:
        if t and t != NO_TYPE:
            multiplier *= chart.get((attack_type, t), 1.0)
    return multiplier


def attacking_types() -> list[str]:
    """All attacking types in the type chart (the 17 Gen 2 types)."""
    return sorted({attack for attack, _ in load_type_chart()})


def build_species_table(battles: list[dict]) -> dict[str, dict]:
    """Return {species: {"types": [...], "base_hp": ..., ...}} from P1 teams and P2 leads.

    Pass training battles only, so the test set is never used to build features.
    A species always has the same types and stats in the data, so the first sighting is kept.
    """
    table = {}
    for battle in battles:
        for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            if p["name"] not in table:
                table[p["name"]] = {"types": list(p["types"]), **{k: p[k] for k in STAT_KEYS}}
    return table


@lru_cache(maxsize=1)
def training_species_table() -> dict[str, dict]:
    """Species table from the training battles, loaded once.

    It holds only fixed facts (types, base stats), never labels, so it is
    safe to use inside cross-validation. It covers 239 species and 99.97% of
    the P2 Pokémon seen in timelines.
    """
    return build_species_table(load_battles("train"))


def types_of(battle: dict, side: str, name: str):
    """Types of Pokémon `name` of `side`, or None if unknown (P2 species missing from the species table)."""
    if side == "p1":
        for p in battle["p1_team_details"]:
            if p["name"] == name:
                return p["types"]
    if name == battle["p2_lead_details"]["name"]:
        return battle["p2_lead_details"]["types"]
    entry = training_species_table().get(name)
    return entry["types"] if entry else None


# ---------------------------------------------------------------- per-battle facts

def latest_states(battle: dict) -> dict[str, dict[str, dict]]:
    """Return {"p1": {name: state}, "p2": {name: state}}: the last recorded state of every Pokémon seen.

    A Pokémon's state only changes while it is on the field, so its last
    recorded state still holds at turn 30.
    """
    latest = {side: {} for side in SIDES}
    for turn in battle["battle_timeline"]:
        for side in SIDES:
            state = turn[f"{side}_pokemon_state"]
            latest[side][state["name"]] = state
    return latest


def hidden_p2_faints(battle: dict, hp_threshold: float = HIDDEN_FAINT_HP) -> dict[str, float]:
    """Return {name: last HP} of P2 Pokémon that probably fainted without it being recorded.

    A P2 Pokémon counts when it left the field at `hp_threshold` HP or less,
    without a recorded faint, and never appeared again in the visible turns.
    """
    timeline = battle["battle_timeline"]
    hidden = {}
    for i in range(1, len(timeline)):
        before, now = timeline[i - 1]["p2_pokemon_state"], timeline[i]["p2_pokemon_state"]
        left_field = before["name"] != now["name"] and before["status"] != "fnt"
        if not left_field or before["hp_pct"] > hp_threshold:
            continue
        if all(t["p2_pokemon_state"]["name"] != before["name"] for t in timeline[i:]):
            hidden[before["name"]] = before["hp_pct"]
    return hidden


def fainted_counts(battle: dict, hp_threshold: float = HIDDEN_FAINT_HP) -> dict[str, int]:
    """Fainted Pokémon per side at turn 30.

    Returns "p1", "p2_recorded", "p2_hidden" and "p2" (= recorded + hidden,
    comparable with P1's count, which is always recorded).
    """
    latest = latest_states(battle)
    p2_recorded = sum(s["status"] == "fnt" for s in latest["p2"].values())
    p2_hidden = len(hidden_p2_faints(battle, hp_threshold))
    return {
        "p1": sum(s["status"] == "fnt" for s in latest["p1"].values()),
        "p2_recorded": p2_recorded,
        "p2_hidden": p2_hidden,
        "p2": p2_recorded + p2_hidden,
    }


def alive_states(battle: dict) -> dict[str, list[dict]]:
    """Last state of every seen Pokémon that has not fainted, per side (P2's hidden faints excluded)."""
    latest = latest_states(battle)
    hidden = hidden_p2_faints(battle)
    return {
        "p1": [s for s in latest["p1"].values() if s["status"] != "fnt"],
        "p2": [s for n, s in latest["p2"].items() if s["status"] != "fnt" and n not in hidden],
    }


def p2_revealed(battle: dict) -> tuple[list[dict], int]:
    """Return (details of P2's revealed Pokémon, number of revealed ones missing from the species table).

    The lead comes first with its own details; the others (seen in the
    timeline) get types and stats from the species table. Missing species are
    skipped and counted.
    """
    table = training_species_table()
    names = [battle["p2_lead_details"]["name"]]
    for turn in battle["battle_timeline"]:
        name = turn["p2_pokemon_state"]["name"]
        if name not in names:
            names.append(name)
    revealed, missing = [battle["p2_lead_details"]], 0
    for name in names[1:]:
        if name in table:
            revealed.append({"name": name, **table[name]})
        else:
            missing += 1
    return revealed, missing


def turn_table(battle: dict) -> list[dict]:
    """One row per turn with what happened to each side.

    For each side ("p1_..." / "p2_..."): team HP left after the turn (in
    "Pokémon worth", 6.0 = all six at full HP, unseen Pokémon count as full),
    damage taken and HP healed this turn, Pokémon lost this turn (P2's hidden
    faints included, at the turn the Pokémon vanished), and whether it switched.
    The KO totals always match `fainted_counts`.
    """
    timeline = battle["battle_timeline"]
    hidden = hidden_p2_faints(battle)
    # Index of each P2 Pokémon's last appearance: a hidden faint happens when it leaves for the last time.
    last_index = {t["p2_pokemon_state"]["name"]: i for i, t in enumerate(timeline)}
    last_hp = {side: {} for side in SIDES}
    fainted = {side: set() for side in SIDES}
    rows = []
    for i, turn in enumerate(timeline):
        row = {"turn": turn["turn"]}
        for side in SIDES:
            state = turn[f"{side}_pokemon_state"]
            name = state["name"]
            prev_name = timeline[i - 1][f"{side}_pokemon_state"]["name"] if i else name
            change = state["hp_pct"] - last_hp[side].get(name, 1.0)
            damage, healed, ko = max(0.0, -change), max(0.0, change), 0

            if state["status"] == "fnt" and name not in fainted[side]:
                fainted[side].add(name)
                ko += 1
            if (side == "p2" and prev_name != name and prev_name in hidden
                    and last_index[prev_name] == i - 1 and prev_name not in fainted[side]):
                fainted[side].add(prev_name)
                damage += last_hp[side].get(prev_name, 0.0)
                last_hp[side][prev_name] = 0.0
                ko += 1

            last_hp[side][name] = state["hp_pct"]
            row[f"{side}_hp_left"] = sum(last_hp[side].values()) + TEAM_SIZE - len(last_hp[side])
            row[f"{side}_damage_taken"] = damage
            row[f"{side}_healed"] = healed
            row[f"{side}_ko_suffered"] = ko
            row[f"{side}_switched"] = int(prev_name != name)
        rows.append(row)
    return rows
