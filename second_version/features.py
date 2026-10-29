"""Features of the second version: the battle's position at turn 30, who is still alive, and P2's hidden side.

On top of the first version's features (fainted Pokémon and team HP with P2's hidden faints,
statuses, active Pokémon, reserves, team strength):
- Spikes: whether each side's Spikes are still up at turn 30 (laid, not cleared by Rapid Spin since).
- Indicators: which species are still alive and which moves each side has used.
- Survivors: type and speed matchups between the Pokémon still alive, their HP-weighted offense and
  bulk, the HP of each surviving species, and the duel between the two active Pokémon.
- P2 team lookup: P2's team is looked up among the P1 teams of the training battles (teams are
  shared across battles); P2's "alive" indicators also count its hidden Pokémon by probability.
- KO detector: a classifier trained on P1's side (where knock-outs are recorded) estimates which of
  P2's Pokémon that left the field for good were knocked out.
Only training battles are used to choose indicator columns, build the lookup library and train the
detector (counts and faint events only, never the battle result).

`build_tables(train_battles, test_battles)` returns the train and test tables with the same columns.
"""

import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from first_version.features import build_table as first_version_table, data_extr  # noqa: E402

# Indicator columns are kept when they appear in at least this many training battles.
MIN_BATTLES = 100


def spikes_features(battle: dict) -> dict:
    """1 if each side's Spikes are still on the opponent's field at turn 30, and the P1-minus-P2 difference."""
    up = {"p1": 0, "p2": 0}
    for turn in battle["battle_timeline"]:
        for side, other in (("p1", "p2"), ("p2", "p1")):
            move = turn[f"{side}_move_details"]
            if move and move["name"] == "spikes":
                up[side] = 1
            elif move and move["name"] == "rapidspin":
                up[other] = 0
    return {"bag_p1_spikes_up": up["p1"], "bag_p2_spikes_up": up["p2"],
            "bag_spikes_up_diff": up["p1"] - up["p2"]}


def indicators(battle: dict) -> dict:
    """1 for each species still alive and each move used, per side.

    P1's unused Pokémon count as alive. For P2 only revealed Pokémon are known
    (hidden faints excluded).
    """
    alive = data_extr.alive_states(battle)
    latest = data_extr.latest_states(battle)
    names = {"p1": {s["name"] for s in alive["p1"]}, "p2": {s["name"] for s in alive["p2"]}}
    names["p1"] |= {p["name"] for p in battle["p1_team_details"] if p["name"] not in latest["p1"]}
    found = {f"bag_{side}_alive_{n}": 1 for side in ("p1", "p2") for n in names[side]}
    for turn in battle["battle_timeline"]:
        for side in ("p1", "p2"):
            move = turn[f"{side}_move_details"]
            if move:
                found[f"bag_{side}_used_{move['name']}"] = 1
    return found


def _base_tables(train_battles: list[dict], test_battles: list[dict]):
    """First-version features + Spikes + alive/moves indicators, same columns for train and test."""
    tables = []
    for battles in (train_battles, test_battles):
        spikes = pd.DataFrame([spikes_features(b) for b in battles])
        bags = pd.DataFrame([indicators(b) for b in battles]).fillna(0).astype(int)
        tables.append((first_version_table(battles), spikes, bags))

    (train_first, train_spikes, train_bags), (test_first, test_spikes, test_bags) = tables
    keep = sorted(c for c in train_bags.columns if train_bags[c].sum() >= MIN_BATTLES)
    train = pd.concat([train_first, train_spikes, train_bags[keep]], axis=1)
    test = pd.concat([test_first, test_spikes, test_bags.reindex(columns=keep, fill_value=0)], axis=1)
    return train, test


SIDES = ("p1", "p2")


def _survivor_hp(battle: dict) -> dict[str, dict[str, float]]:
    """{side: {species: HP}} of surviving Pokémon (P1's unused ones at 1.0; P2 revealed only)."""
    alive = data_extr.alive_states(battle)
    latest = data_extr.latest_states(battle)
    hp = {side: {s["name"]: s["hp_pct"] for s in alive[side]} for side in SIDES}
    hp["p1"].update({p["name"]: 1.0 for p in battle["p1_team_details"] if p["name"] not in latest["p1"]})
    return hp


def _details(battle: dict, side: str, name: str):
    """Types and base stats of one Pokémon (P1 team, P2 lead, or species table), or None."""
    if side == "p1":
        return next((p for p in battle["p1_team_details"] if p["name"] == name), None)
    if name == battle["p2_lead_details"]["name"]:
        return battle["p2_lead_details"]
    return data_extr.training_species_table().get(name)


def _best_hit(attacker: dict, defender: dict) -> float:
    own = [t for t in attacker["types"] if t != data_extr.NO_TYPE]
    return max((data_extr.effectiveness(t, defender["types"]) for t in own), default=1.0)


def _shown_moves(battle: dict) -> dict[str, dict[str, set]]:
    """{side: {species: moves used}}, skipping switch-in turns where the move may belong to the Pokémon that left."""
    timeline = battle["battle_timeline"]
    shown = {side: defaultdict(set) for side in SIDES}
    for i, turn in enumerate(timeline):
        for side in SIDES:
            name = turn[f"{side}_pokemon_state"]["name"]
            move = turn[f"{side}_move_details"]
            if move and (i == 0 or timeline[i - 1][f"{side}_pokemon_state"]["name"] == name):
                shown[side][name].add(move["name"])
    return shown


def survivor_features(battle: dict) -> dict[str, float]:
    """Matchups, speed and HP-weighted offense/bulk of each side's survivors."""
    hp = _survivor_hp(battle)
    surv = {side: [(d, h) for n, h in hp[side].items() if (d := _details(battle, side, n)) is not None] for side in SIDES}
    mean = lambda v: sum(v) / len(v) if v else 0.0
    p1, p2 = [d for d, _ in surv["p1"]], [d for d, _ in surv["p2"]]
    f = {
        "surv_p1_share_se": mean([_best_hit(a, d) >= 2 for a in p1 for d in p2]),
        "surv_p2_share_se": mean([_best_hit(a, d) >= 2 for a in p2 for d in p1]),
        "surv_p2_threatened": mean([any(_best_hit(a, d) >= 2 for a in p1) for d in p2]),
        "surv_p1_threatened": mean([any(_best_hit(a, d) >= 2 for a in p2) for d in p1]),
    }
    f["surv_share_se_diff"] = f["surv_p1_share_se"] - f["surv_p2_share_se"]
    f["surv_threatened_diff"] = f["surv_p2_threatened"] - f["surv_p1_threatened"]
    speeds = {"p1": [d["base_spe"] for d in p1], "p2": [d["base_spe"] for d in p2]}
    for side in SIDES:
        f[f"surv_{side}_max_spe"] = max(speeds[side], default=0)
        f[f"surv_{side}_mean_spe"] = mean(speeds[side])
    f["surv_max_spe_diff"] = f["surv_p1_max_spe"] - f["surv_p2_max_spe"]
    f["surv_p1_n_faster"] = sum(v > f["surv_p2_max_spe"] for v in speeds["p1"])
    for side in SIDES:
        f[f"fuel_{side}_offense"] = sum(h * max(d["base_atk"], d["base_spa"]) for d, h in surv[side])
        f[f"fuel_{side}_bulk"] = sum(h * (d["base_hp"] + d["base_def"] + d["base_spd"]) for d, h in surv[side])
    f["fuel_offense_diff"] = f["fuel_p1_offense"] - f["fuel_p2_offense"]
    f["fuel_bulk_diff"] = f["fuel_p1_bulk"] - f["fuel_p2_bulk"]
    return f


def species_hp(battle: dict) -> dict[str, float]:
    """HP of each surviving species, per side."""
    return {f"hp_{side}_{name}": v for side, d in _survivor_hp(battle).items() for name, v in d.items()}


def duel_features(battle: dict) -> dict[str, float]:
    """The two active Pokémon at turn 30: type advantage (own types and shown attacks), speed, fainted."""
    last = battle["battle_timeline"][-1]
    shown = _shown_moves(battle)
    move_types = {t[f"{s}_move_details"]["name"]: t[f"{s}_move_details"]["type"]
                  for t in battle["battle_timeline"] for s in SIDES
                  if t[f"{s}_move_details"] and t[f"{s}_move_details"]["category"] != "status"
                  and t[f"{s}_move_details"]["name"] != "hiddenpower"}
    states = {side: last[f"{side}_pokemon_state"] for side in SIDES}
    types = {side: data_extr.types_of(battle, side, states[side]["name"]) for side in SIDES}
    f = {}
    for side, other in (("p1", "p2"), ("p2", "p1")):
        f[f"duel_{side}_active_fainted"] = int(states[side]["status"] == "fnt")
        if types[side] is None or types[other] is None:
            f[f"duel_{side}_type_hit"] = f[f"duel_{side}_move_hit"] = 1.0
            continue
        own = [t for t in types[side] if t != data_extr.NO_TYPE]
        f[f"duel_{side}_type_hit"] = max((data_extr.effectiveness(t, types[other]) for t in own), default=1.0)
        moves = [move_types[m] for m in shown[side].get(states[side]["name"], ()) if m in move_types]
        f[f"duel_{side}_move_hit"] = max((data_extr.effectiveness(t, types[other]) for t in moves), default=1.0)
    f["duel_type_hit_diff"] = f["duel_p1_type_hit"] - f["duel_p2_type_hit"]
    f["duel_move_hit_diff"] = f["duel_p1_move_hit"] - f["duel_p2_move_hit"]
    stage = lambda s: (2 + s) / 2 if s >= 0 else 2 / (2 - s)
    speed = {}
    for side in SIDES:
        d = _details(battle, side, states[side]["name"])
        v = (d["base_spe"] if d else 0) * stage(states[side]["boosts"].get("spe", 0))
        speed[side] = v / 4 if states[side]["status"] == "par" else v
    f["duel_p1_speed"], f["duel_p2_speed"] = speed["p1"], speed["p2"]
    f["duel_p1_faster"] = int(speed["p1"] > speed["p2"]) - int(speed["p2"] > speed["p1"])
    return f


@lru_cache(maxsize=1)
def _team_library():
    """Counts of P1 teams (as species sets) in the training battles, and an index species -> teams."""
    counts = Counter(frozenset(p["name"] for p in b["p1_team_details"]) for b in data_extr.load_battles("train"))
    index = defaultdict(set)
    for team in counts:
        for name in team:
            index[name].add(team)
    return counts, index


def _p2_lookup(battle: dict):
    """(revealed P2 species, {candidate team: probability}) from the library teams containing all revealed ones."""
    counts, index = _team_library()
    seen = set(data_extr.latest_states(battle)["p2"]) | {battle["p2_lead_details"]["name"]}
    teams = set.intersection(*(index.get(n, set()) for n in seen))
    total = sum(counts[t] for t in teams)
    return seen, ({t: counts[t] / total for t in teams} if total else {})


def lookup_features(battle: dict) -> dict[str, float]:
    """How well P2's team is identified, and the expected stats of its hidden Pokémon."""
    table = data_extr.training_species_table()
    seen, cands = _p2_lookup(battle)
    hidden = Counter()
    for team, p in cands.items():
        for name in team - seen:
            hidden[name] += p
    norm = sum(hidden.values())
    expect = lambda value: sum(p * value(table[s]) for s, p in hidden.items() if s in table) / norm if norm else 0.0
    f = {"look_n_hidden": data_extr.TEAM_SIZE - len(seen), "look_n_candidates": min(len(cands), 50),
         "look_found": int(bool(cands)), "look_unique": int(len(cands) == 1),
         "look_top_share": max(cands.values(), default=0.0)}
    for key in ("base_atk", "base_spa", "base_spe", "base_hp"):
        f[f"look_hidden_exp_{key.removeprefix('base_')}"] = expect(lambda d, k=key: d[k])
    f["look_hidden_exp_total"] = expect(lambda d: sum(d[k] for k in data_extr.STAT_KEYS))
    return f


def p2_full_team(battle: dict) -> dict[str, float]:
    """P2's alive indicators for its whole team: revealed survivors count 1, hidden Pokémon their lookup probability."""
    seen, cands = _p2_lookup(battle)
    f = {f"bag_p2_full_{s['name']}": 1.0 for s in data_extr.alive_states(battle)["p2"]}
    for team, p in cands.items():
        for name in team - seen:
            f[f"bag_p2_full_{name}"] = f.get(f"bag_p2_full_{name}", 0.0) + p
    return f


def _leave_clues(battle: dict, side: str, i: int) -> dict:
    """Clues about why `side`'s active Pokémon at turn i-1 is gone at the end of turn i."""
    timeline = battle["battle_timeline"]
    other = data_extr.OTHER_SIDE[side]
    before = timeline[i - 1][f"{side}_pokemon_state"]
    opp, own = timeline[i][f"{other}_move_details"], timeline[i][f"{side}_move_details"]
    types = data_extr.types_of(battle, side, before["name"]) or []
    boosts = timeline[i - 1][f"{other}_pokemon_state"]["boosts"]
    damaging = bool(opp and opp["category"] != "status")
    return {"hp_before": before["hp_pct"], "turn": i + 1,
            "status_dot": int(before["status"] in ("psn", "tox", "brn")), "asleep": int(before["status"] == "slp"),
            "opp_damaging": int(damaging), "opp_power": opp["base_power"] if damaging else 0,
            "opp_effect": data_extr.effectiveness(opp["type"], types) if damaging and types else 1.0,
            "opp_physical": int(damaging and opp["category"] == "physical"),
            "opp_phaze": int(bool(opp and opp["name"] in ("roar", "whirlwind"))),
            "opp_priority": opp["priority"] if opp else 0, "own_moved": int(own is not None),
            "own_self_ko": int(bool(own and own["name"] in ("explosion", "selfdestruct", "destinybond"))),
            "opp_atk_boost": boosts.get("atk", 0) + boosts.get("spa", 0)}


@lru_cache(maxsize=1)
def _ko_detector():
    """Classifier of "fainted vs switched out", trained on P1's leave events in the training battles."""
    rows, labels = [], []
    for battle in data_extr.load_battles("train"):
        timeline = battle["battle_timeline"]
        for i in range(1, len(timeline)):
            before, now = timeline[i - 1]["p1_pokemon_state"], timeline[i]["p1_pokemon_state"]
            if before["status"] == "fnt":
                continue
            if now["name"] == before["name"] and now["status"] == "fnt":
                rows.append(_leave_clues(battle, "p1", i)); labels.append(1)
            elif now["name"] != before["name"]:
                rows.append(_leave_clues(battle, "p1", i)); labels.append(0)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                           random_state=0, verbose=-1)
    return model.fit(pd.DataFrame(rows), np.array(labels))


def detector_features(battle: dict) -> dict[str, float]:
    """P2's knock-outs and HP corrected with the detector's probabilities (soft counts), and diffs with P1."""
    timeline = battle["battle_timeline"]
    last_index = {t["p2_pokemon_state"]["name"]: i for i, t in enumerate(timeline)}
    events, names = [], []
    for i in range(1, len(timeline)):
        before, now = timeline[i - 1]["p2_pokemon_state"], timeline[i]["p2_pokemon_state"]
        if before["status"] != "fnt" and now["name"] != before["name"] and last_index[before["name"]] == i - 1:
            events.append(_leave_clues(battle, "p2", i)); names.append(before["name"])
    probs = dict(zip(names, _ko_detector().predict_proba(pd.DataFrame(events))[:, 1])) if events else {}
    latest = data_extr.latest_states(battle)
    p1_fnt = sum(s["status"] == "fnt" for s in latest["p1"].values())
    soft = sum(s["status"] == "fnt" for s in latest["p2"].values()) + sum(probs.values())
    p2_hp = sum(s["hp_pct"] for s in latest["p2"].values()) + data_extr.TEAM_SIZE - len(latest["p2"])
    p2_hp_soft = p2_hp - sum(p * latest["p2"][n]["hp_pct"] for n, p in probs.items())
    p1_hp = sum(s["hp_pct"] for s in latest["p1"].values()) + data_extr.TEAM_SIZE - len(latest["p1"])
    return {"det_p2_hidden_soft": sum(probs.values()), "det_p2_fainted_soft": soft,
            "det_fainted_diff_soft": p1_fnt - soft, "det_p2_hp_left_soft": p2_hp_soft,
            "det_hp_left_diff_soft": p1_hp - p2_hp_soft}


def _indicator_tables(train_battles, test_battles, func):
    """Same-column tables of `func` for train and test, keeping columns non-zero in >= MIN_BATTLES training battles."""
    train = pd.DataFrame([func(b) for b in train_battles]).fillna(0)
    keep = sorted(c for c in train.columns if (train[c] != 0).sum() >= MIN_BATTLES)
    test = pd.DataFrame([func(b) for b in test_battles]).fillna(0).reindex(columns=keep, fill_value=0)
    return train[keep], test


def build_tables(train_battles: list[dict], test_battles: list[dict]):
    """Train and test tables: `battle_id`, all features, and `player_won` for train."""
    second_train, second_test = _base_tables(train_battles, test_battles)
    full_train, full_test = _indicator_tables(train_battles, test_battles, p2_full_team)
    hp_train, hp_test = _indicator_tables(train_battles, test_battles, species_hp)
    out = []
    for second, full, hp, battles in ((second_train, full_train, hp_train, train_battles),
                                      (second_test, full_test, hp_test, test_battles)):
        # P2's alive indicators are replaced by the full-team version (hidden Pokémon by probability).
        second = second.drop(columns=[c for c in second.columns if c.startswith("bag_p2_alive_")])
        extra = [pd.DataFrame([f(b) for b in battles]) for f in (lookup_features, detector_features)]
        surv = pd.DataFrame([survivor_features(b) for b in battles])
        duel = pd.DataFrame([duel_features(b) for b in battles])
        out.append(pd.concat([second, full] + extra + [surv, hp, duel], axis=1))
    return out[0], out[1]
