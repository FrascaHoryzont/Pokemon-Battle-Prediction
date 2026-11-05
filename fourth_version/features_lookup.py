"""Soft P2 team reconstruction and enhanced KO detector for Version 3.

Features:
- Unsupervised team library using train + test P1 teams
- Co-occurrence matrix P(species_B | species_A)
- Soft Jaccard fallback when exact team intersection fails
- Enhanced KO detector with Spikes awareness and self-destruct detection
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import data_extr  # noqa: E402

def build_team_library(train_battles: list[dict], n_top_species: int = 10) -> dict:
    """Builds team library, co-occurrence matrix, and top meta species strictly from training P1 teams."""
    counts = Counter(frozenset(p["name"] for p in b["p1_team_details"]) for b in train_battles)
    index = defaultdict(set)
    species_counts = Counter()

    for team in counts:
        for name in team:
            index[name].add(team)
            species_counts[name] += counts[team]

    top_species = [sp for sp, _ in species_counts.most_common(n_top_species)]

    # Co-occurrence: P(s2 | s1) = count(s1 and s2) / count(s1)
    cooccurrence = defaultdict(dict)
    all_species = list(species_counts.keys())
    for s1 in all_species:
        s1_teams = index[s1]
        denom = species_counts[s1]
        for s2 in all_species:
            if s1 == s2:
                cooccurrence[s1][s2] = 1.0
            else:
                num = sum(counts[t] for t in s1_teams if s2 in t)
                cooccurrence[s1][s2] = num / denom if denom > 0 else 0.0

    return {
        "counts": counts,
        "index": index,
        "species_counts": species_counts,
        "cooccurrence": cooccurrence,
        "top_species": top_species,
    }


_DEFAULT_LIBRARY = None


def default_team_library() -> dict:
    """Builds or returns the shared team library strictly from training data."""
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        train_battles = data_extr.load_battles("train")
        _DEFAULT_LIBRARY = build_team_library(train_battles)
    return _DEFAULT_LIBRARY


def soft_lookup_features(battle: dict, library: dict = None) -> dict[str, float]:
    """Soft probabilistic lookup of P2's hidden roster with Jaccard and co-occurrence fallbacks."""
    lib = library if library is not None else default_team_library()
    counts = lib["counts"]
    index = lib["index"]
    coocc = lib["cooccurrence"]
    table = data_extr.training_species_table()

    seen = set(data_extr.latest_states(battle)["p2"]) | {battle["p2_lead_details"]["name"]}
    exact_candidates = set.intersection(*(index.get(n, set()) for n in seen)) if seen else set()

    cands_weighted = Counter()
    is_exact = 0.0

    if exact_candidates:
        is_exact = 1.0
        for t in exact_candidates:
            cands_weighted[t] = counts[t]
    else:
        # Soft fallback: teams that share at least max(1, len(seen)-1) species
        min_overlap = max(1, len(seen) - 1)
        for s in seen:
            for t in index.get(s, set()):
                overlap = len(t & seen)
                if overlap >= min_overlap:
                    jaccard = overlap / len(t | seen)
                    cands_weighted[t] = max(cands_weighted[t], jaccard * counts[t])

    total_w = sum(cands_weighted.values())
    hidden_probs = Counter()

    if total_w > 0:
        for team, w in cands_weighted.items():
            prob = w / total_w
            for sp in team - seen:
                hidden_probs[sp] += prob
    else:
        # Ultimate fallback: Bayesian co-occurrence combination
        for sp in table:
            if sp not in seen:
                # 1 - prod(1 - P(sp | s))
                not_prob = 1.0
                for s in seen:
                    p = coocc.get(s, {}).get(sp, 0.02)
                    not_prob *= (1.0 - p)
                hidden_probs[sp] = 1.0 - not_prob

    norm = sum(hidden_probs.values())
    expect = lambda value: sum(p * value(table[s]) for s, p in hidden_probs.items() if s in table) / norm if norm > 0 else 0.0

    f = {
        "look_n_hidden": float(data_extr.TEAM_SIZE - len(seen)),
        "look_found_exact": is_exact,
        "look_n_candidates": float(min(len(cands_weighted), 50)),
        "look_top_share": max((w / total_w for w in cands_weighted.values()), default=0.0) if total_w > 0 else 0.0
    }

    # Meta species probabilities for P2
    for sp in lib.get("top_species", []):
        if sp in seen:
            f[f"look_p2_prob_{sp}"] = 1.0
        else:
            f[f"look_p2_prob_{sp}"] = min(1.0, hidden_probs.get(sp, 0.0))

    # Expected hidden stats
    for key in ("base_atk", "base_spa", "base_spe", "base_hp"):
        f[f"look_hidden_exp_{key.removeprefix('base_')}"] = expect(lambda d, k=key: d[k])
    f["look_hidden_exp_total"] = expect(lambda d: sum(d[k] for k in data_extr.STAT_KEYS))

    return f


def p2_full_team_distribution(battle: dict, library: dict = None) -> dict[str, float]:
    """Returns probability of each species being in P2's full 6-member team."""
    lib = library if library is not None else default_team_library()
    seen = set(data_extr.latest_states(battle)["p2"]) | {battle["p2_lead_details"]["name"]}
    soft_f = soft_lookup_features(battle, lib)
    f = {}
    for sp in lib.get("top_species", []):
        f[f"p2_full_prob_{sp}"] = 1.0 if sp in seen else soft_f.get(f"look_p2_prob_{sp}", 0.0)
    return f


def _enhanced_leave_clues(battle: dict, side: str, i: int) -> dict:
    """Clues about why `side`'s active Pokémon left the field at turn i, including Spikes and self-sacrifice."""
    timeline = battle["battle_timeline"]
    other = data_extr.OTHER_SIDE[side]
    before = timeline[i - 1][f"{side}_pokemon_state"]
    opp, own = timeline[i][f"{other}_move_details"], timeline[i][f"{side}_move_details"]
    types = data_extr.types_of(battle, side, before["name"]) or []
    boosts = timeline[i - 1][f"{other}_pokemon_state"]["boosts"]
    damaging = bool(opp and opp.get("category") != "status")

    # Spikes active on side's field at turn i
    spikes_on_side = 0.0
    for t in timeline[:i]:
        m = t[f"{other}_move_details"]
        own_m = t[f"{side}_move_details"]
        if m and m["name"] == "spikes":
            spikes_on_side = 1.0
        elif own_m and own_m["name"] == "rapidspin":
            spikes_on_side = 0.0

    self_ko = int(bool(own and own.get("name") in ("explosion", "selfdestruct", "destinybond")))

    return {
        "hp_before": before["hp_pct"],
        "turn": i + 1,
        "status_dot": int(before["status"] in ("psn", "tox", "brn")),
        "asleep": int(before["status"] == "slp"),
        "opp_damaging": int(damaging),
        "opp_power": opp.get("base_power", 0) if damaging else 0,
        "opp_effect": data_extr.effectiveness(opp.get("type", "normal"), types) if damaging and types else 1.0,
        "opp_physical": int(damaging and opp.get("category") == "physical"),
        "opp_phaze": int(bool(opp and opp.get("name") in ("roar", "whirlwind"))),
        "opp_priority": opp.get("priority", 0) if opp else 0,
        "own_moved": int(own is not None),
        "own_self_ko": self_ko,
        "spikes_on_side": spikes_on_side,
        "opp_atk_boost": boosts.get("atk", 0) + boosts.get("spa", 0),
    }


_DETECTOR_MODEL = None


def _enhanced_ko_detector():
    """Classifier of 'fainted vs switched out' trained on P1's leave events in training battles."""
    global _DETECTOR_MODEL
    if _DETECTOR_MODEL is not None:
        return _DETECTOR_MODEL

    rows, labels = [], []
    for battle in data_extr.load_battles("train"):
        timeline = battle["battle_timeline"]
        for i in range(1, len(timeline)):
            before, now = timeline[i - 1]["p1_pokemon_state"], timeline[i]["p1_pokemon_state"]
            if before["status"] == "fnt":
                continue
            if now["name"] == before["name"] and now["status"] == "fnt":
                rows.append(_enhanced_leave_clues(battle, "p1", i))
                labels.append(1)
            elif now["name"] != before["name"]:
                rows.append(_enhanced_leave_clues(battle, "p1", i))
                labels.append(0)
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
        random_state=42, verbose=-1, n_jobs=-1
    )
    _DETECTOR_MODEL = model.fit(pd.DataFrame(rows), np.array(labels))
    return _DETECTOR_MODEL


def enhanced_detector_features(battle: dict) -> dict[str, float]:
    """P2's knock-outs and HP corrected with the enhanced detector's probabilities."""
    timeline = battle["battle_timeline"]
    last_index = {t["p2_pokemon_state"]["name"]: i for i, t in enumerate(timeline)}
    events, names = [], []
    for i in range(1, len(timeline)):
        before, now = timeline[i - 1]["p2_pokemon_state"], timeline[i]["p2_pokemon_state"]
        if before["status"] != "fnt" and now["name"] != before["name"] and last_index[before["name"]] == i - 1:
            events.append(_enhanced_leave_clues(battle, "p2", i))
            names.append(before["name"])

    probs = dict(zip(names, _enhanced_ko_detector().predict_proba(pd.DataFrame(events))[:, 1])) if events else {}
    latest = data_extr.latest_states(battle)
    p1_fnt = sum(s["status"] == "fnt" for s in latest["p1"].values())
    soft = sum(s["status"] == "fnt" for s in latest["p2"].values()) + sum(probs.values())
    p2_hp = sum(s["hp_pct"] for s in latest["p2"].values()) + data_extr.TEAM_SIZE - len(latest["p2"])
    p2_hp_soft = p2_hp - sum(p * latest["p2"][n]["hp_pct"] for n, p in probs.items())
    p1_hp = sum(s["hp_pct"] for s in latest["p1"].values()) + data_extr.TEAM_SIZE - len(latest["p1"])

    return {
        "det_p2_hidden_soft": sum(probs.values()),
        "det_p2_fainted_soft": soft,
        "det_fainted_diff_soft": p1_fnt - soft,
        "det_p2_hp_left_soft": p2_hp_soft,
        "det_hp_left_diff_soft": p1_hp - p2_hp_soft
    }
