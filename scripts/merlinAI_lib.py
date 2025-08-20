from __future__ import annotations

from scipy.stats import truncnorm

import numpy as np

import random

import yaml

from pathlib import Path

_EPS = 1e-12


CANONICAL_CARD_TYPES = [
    "creature",
    "artifact creature",
    "planeswalker",
    "instant",
    "sorcery",
    "enchantment",
    "saga",
    "battle",
    "land",
    "basic land",
    "artifact",
    "kindred",
]
DEFAULT_TYPE_WEIGHTS = {
    "creature": 50,
    "artifact creature": 0,
    "planeswalker": 2,
    "instant": 12,
    "sorcery": 12,
    "enchantment": 12,
    "saga": 0,
    "battle": 0,
    "land": 12,
    "basic land": 0,
    "artifact": 0,
    "kindred": 0,
}
# --- canonical color order (W U B R G, then colorless) ---
CANONICAL_COLOR_ORDER = ["white", "blue", "black", "red", "green", "colorless"]


STRICT = True 

def truncated_normal_random(mean: float, sd=0.35, low=0.0, high=1.0):
    """
    Draw a random float from a truncated normal distribution.

    mean : center of the normal distribution
    sd   : standard deviation (spread of values)
    low  : lower bound
    high : upper bound
    """
    if truncnorm is None:
        raise RuntimeError(
            "scipy is required for truncated_normal_random; "
            "install with `pip install scipy`."
        )

    # Compute a, b for truncnorm (in standard normal space)
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd)


def beta_skewed_random(low: float, high: float, *,
                       skew: int = 0,
                       rng: np.random.Generator | None = None) -> float:
    """
    General bounded sampler on [low, high] using Beta(alpha, beta).

    skew ∈ {-2, -1, 0, 1, 2}:
        -2  -> strong skew toward LOW
        -1  -> slight skew toward LOW
         0  -> balanced (symmetric)
        +1  -> slight skew toward HIGH
        +2  -> strong skew toward HIGH

    Internally uses simple, readable (alpha, beta) pairs that sum to a
    constant "concentration" (tighter than uniform but not too spiky).
    Tweak ALPHA_BETA_MAP if you want different sharpness.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert low < high, "low must be < high"

    # Sum(alpha,beta) ~= concentration; larger => tighter peak.
    # These are sensible defaults; adjust if you want stronger/weaker skew.
    ALPHA_BETA_MAP = {
        -2: (2.0, 6.0),  # strong low
        -1: (3.0, 5.0),  # slight  low
         0: (4.0, 4.0),  # balanced
         1: (5.0, 3.0),  # slight  high
         2: (6.0, 2.0),  # strong high
    }
    if skew not in ALPHA_BETA_MAP:
        raise ValueError("skew must be one of {-2, -1, 0, 1, 2}")

    alpha, beta = ALPHA_BETA_MAP[skew]
    # Sample on [0,1] then scale
    x01 = rng.beta(alpha, beta)
    return float(low + x01 * (high - low))

def check_mutation(mutation_chance) -> bool:
    """
    Check if a mutation occurs based on the given chance.
    """
    d100 = random.random() * 100

    return d100 <= mutation_chance

# ========= CLI: Check & Normalize Config =========

def check_and_normalize_config(config_path: str, save: bool = False, total: float = 100.0):
    """
    User-facing CLI helper to check + normalize weights in a config.yaml.

    - Normalizes colors_weights (dict or list), rarities_weights (dict or list),
      and card_types_weights rows (dict or list) to sum `total`.
    - If lists are provided where dicts are preferred, converts them to dicts using labels.
    - Prints detailed diffs (before -> after).
    - Optionally saves the normalized config back to file.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"❌ Config file not found: {path}")
        return

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    fixed = _normalize_all_weights_with_diffs(config, total=total)

    if save:
        with open(path, "w") as f:
            yaml.safe_dump(fixed, f, sort_keys=False)
        print(f"\n✅ Saved normalized config back to {path}")
    else:
        print("\nℹ️  Normalization complete (not saved). Use --save to overwrite the file.")
    return fixed


def _ordered_color_keys(keys: list[str] | set[str]) -> list[str]:
    order_index = {c: i for i, c in enumerate(CANONICAL_COLOR_ORDER)}
    keys = list(keys)
    known = [k for k in keys if k in order_index]
    known.sort(key=lambda k: order_index[k])
    others = [k for k in keys if k not in order_index]
    others.sort()
    return known + others


def _reorder_color_dict(d: dict) -> dict:
    """Return a new dict with keys in WUBRG(+colorless) order, then any extras alphabetically."""
    ordered = {}
    for k in _ordered_color_keys(d.keys()):
        ordered[k] = d[k]
    return ordered

def _normalize_all_weights_with_diffs(config: dict, total: float = 100.0) -> dict:
    """
    Walk through config and normalize known weight sections:
      - skeleton_params.colors_weights   (dict preferred; list accepted)
      - skeleton_params.rarities_weights (dict preferred; list accepted)
      - skeleton_params.card_types_weights (dict: per-color rows; list accepted)
        including optional `_default` row
    Uses labels from:
      - skeleton_params.colors
      - skeleton_params.rarities
      - skeleton_params.card_types
    """
    sp = config.get("skeleton_params", {})
    if not isinstance(sp, dict):
        print("⚠️  'skeleton_params' missing or not a dict; nothing to do.")
        return config

    colors = sp.get("colors", [])
    rarities = sp.get("rarities", [])
    card_types = sp.get("card_types")

    if not card_types:
        # Try to derive from weights; fall back to canonical if nothing found
        ctw = sp.get("card_types_weights", {})
        derived = _derive_card_types(ctw) if isinstance(ctw, dict) else []
        card_types = derived or CANONICAL_CARD_TYPES
        sp["card_types"] = list(card_types)
        print("ℹ️  Inserted 'skeleton_params.card_types' into config (derived or canonical).")

    # ---- colors_weights (dict preferred; list accepted) ----
    if "colors_weights" in sp:
        cw = sp["colors_weights"]
        if isinstance(cw, list):
            if not colors:
                print("⚠️  colors_weights is a list but 'colors' is missing; cannot label — leaving as list.")
                sp["colors_weights"] = _fix_length_and_normalize_list(
                    key="skeleton_params.colors_weights", lst=cw, labels=None, total=total
                )
            else:
                print("ℹ️  Converting colors_weights list -> dict using 'colors' labels.")
                labeled = _list_to_labeled_dict(cw, colors)
                sp["colors_weights"] = _normalize_dict_with_diffs(
                    key="skeleton_params.colors_weights", d=labeled, total=total
                )
                
        elif isinstance(cw, dict):
            allowed = set(colors) if colors else None
            cw = _handle_unknown_keys(cw, allowed, "skeleton_params.colors_weights", noun="colors")
            normalized = _normalize_dict_with_diffs(
                key="skeleton_params.colors_weights", d=cw, total=total
            )
            # ensure WUBRG order on save
            sp["colors_weights"] = _reorder_color_dict(normalized)
        else:
            print("ℹ️  skeleton_params.colors_weights is neither list nor dict; skipping.")

    # ---- rarities_weights (dict preferred; list accepted) ----
    if "rarities_weights" in sp:
        rw = sp["rarities_weights"]
        if isinstance(rw, list):
            if not rarities:
                print("⚠️  rarities_weights is a list but 'rarities' is missing; cannot label — leaving as list.")
                sp["rarities_weights"] = _fix_length_and_normalize_list(
                    key="skeleton_params.rarities_weights", lst=rw, labels=None, total=total
                )
            else:
                print("ℹ️  Converting rarities_weights list -> dict using 'rarities' labels.")
                labeled = _list_to_labeled_dict(rw, rarities)
                sp["rarities_weights"] = _normalize_dict_with_diffs(
                    key="skeleton_params.rarities_weights", d=labeled, total=total
                )
        elif isinstance(rw, dict):
            allowed = set(rarities) if rarities else None
            rw = _handle_unknown_keys(rw, allowed, "skeleton_params.rarities_weights", noun="rarities")
            sp["rarities_weights"] = _normalize_dict_with_diffs(
                 key="skeleton_params.rarities_weights", d=rw, total=total
             )

        else:
            print("ℹ️  skeleton_params.rarities_weights is neither list nor dict; skipping.")

    # ---- card_types_weights (dict of rows; list rows accepted) ----
    # ---- card_types_weights (resolve against _default, then normalize) ----
    if "card_types_weights" in sp and isinstance(sp["card_types_weights"], dict):
        ctw: dict = sp["card_types_weights"]

        # 1) Build/normalize the baseline default_map first
        #    Source priority: YAML _default (dict/list) → DEFAULT_TYPE_WEIGHTS
        #    Then restrict to current card_types and fill missing from code defaults
        if isinstance(ctw.get("_default"), list):
            default_map_raw = _list_to_labeled_dict(ctw["_default"], card_types)
        elif isinstance(ctw.get("_default"), dict):
            default_map_raw = dict(ctw["_default"])
        else:
            default_map_raw = {t: DEFAULT_TYPE_WEIGHTS.get(t, 0.0) for t in card_types}

        # Coerce to numeric, keep only declared card_types, fill missing from code defaults
        default_map_coerced = {}
        for t in card_types:
            v = default_map_raw.get(t, DEFAULT_TYPE_WEIGHTS.get(t, 0.0))
            try:
                default_map_coerced[t] = float(v)
            except Exception:
                print(f"⚠️  Skipping non-numeric default for type {t!r}: {v!r}")
                default_map_coerced[t] = float(DEFAULT_TYPE_WEIGHTS.get(t, 0.0))

        # Normalize default row and write it back (so YAML has the clean baseline)
        default_norm = _normalize_dict_with_diffs(
            key="skeleton_params.card_types_weights[_default]",
            d=default_map_coerced,
            total=total,
        )
        ctw["_default"] = default_norm
        # 2) For each other color row (in preferred order):
        #    - convert list→dict if needed
        #    - keep only declared card_types (or drop unknowns if STRICT)
        #    - merge over normalized default (fill missing)
        #    - normalize the merged row

        # Build preferred row order: _default, WUBRG(+colorless), then others
        ordered_rows = ["_default"]
        ordered_rows += [c for c in CANONICAL_COLOR_ORDER if c in ctw]
        seen = set(ordered_rows)
        ordered_rows += sorted([k for k in ctw.keys() if k not in seen])

        # Process in that order (skipping _default here; it's done above)
        for color_key in ordered_rows:
            if color_key == "_default":
                continue
            row = ctw.get(color_key)
            if row is None:
                continue
            if not isinstance(row, (list, dict)):
                print(f"ℹ️  Skip non-weight entry: skeleton_params.card_types_weights[{color_key}]")
                continue

            # Convert list → labeled dict if needed
            if isinstance(row, list):
                print(f"ℹ️  Converting {color_key} row list -> dict using 'card_types' labels.")
                row_map = _list_to_labeled_dict(row, card_types)
            else:
                row_map = dict(row)

            # Handle unknown type keys according to STRICT policy (but allow all in _default)
            row_map = _handle_unknown_keys(
                row_map,
                set(card_types) if card_types else None,
                f"skeleton_params.card_types_weights[{color_key}]",
                noun="types",
            )

            # Coerce numerics; ignore non-numerics with a warning
            row_num: dict[str, float] = {}
            for t, v in row_map.items():
                if t not in card_types:
                    continue
                try:
                    row_num[t] = float(v)
                except Exception:
                    print(
                        f"⚠️  Skipping non-numeric value for "
                        f"skeleton_params.card_types_weights[{color_key}][{t!r}]: {v!r}"
                    )

            # Merge over default baseline, fill missing types from default
            merged = dict(default_norm)
            for t in row_num.keys() & set(card_types):
                merged[t] = row_num[t]

            # Normalize the merged full row and write it back
            ctw[color_key] = _normalize_dict_with_diffs(
                key=f"skeleton_params.card_types_weights[{color_key}] (resolved over _default)",
                d=merged,
                total=total,
            )

        # Rebuild dict in preferred order for nicer YAML output
        sp["card_types_weights"] = {k: ctw[k] for k in ordered_rows if k in ctw}


    return config


# ---------- helpers ----------
def _handle_unknown_keys(d: dict, allowed: set | None, where: str, noun: str = "keys") -> dict:
    """
    If `allowed` is None → return `d` unchanged and do not warn.
    If `allowed` is provided:
      - STRICT = False: warn but keep all keys (no data loss).
      - STRICT = True:  drop unknown keys and warn explicitly.
    """
    if allowed is None:
        return d
    unknown = set(d.keys()) - allowed
    if not unknown:
        return d
    if STRICT:
        print(f"⚠️  {where}: dropping unknown {noun}: {sorted(unknown)}")
        return {k: v for k, v in d.items() if k in allowed}
    else:
        print(f"⚠️  {where}: found {noun} outside the allowed set; keeping them: {sorted(unknown)}")
        return d

def _list_to_labeled_dict(values: list, labels: list[str]) -> dict:
    """Zip list values to labels; truncate or pad with zeros to match label length."""
    n = len(labels)
    vals = (values[:n] + [0.0] * max(0, n - len(values))) if n else list(values)
    return {labels[i]: float(vals[i]) for i in range(n)}

def _fix_length_and_normalize_list(key: str, lst, labels: list[str] | None, total: float):
    """Pad/truncate to match labels length (if provided), then normalize and print diffs."""
    if not isinstance(lst, list):
        print(f"ℹ️  {key} is not a list, skipping length check and normalization.")
        return lst

    original = list(lst)

    if labels is not None and len(labels) > 0:
        n_target = len(labels)
        n = len(lst)
        if n != n_target:
            if n > n_target:
                print(f"⚠️  {key} has length {n} > {n_target} (labels). Truncating extra entries.")
                lst = lst[:n_target]
            else:
                print(f"⚠️  {key} has length {n} < {n_target} (labels). Padding with zeros.")
                lst = lst + [0.0] * (n_target - n)

    s = sum(lst)
    if s == 0:
        print(f"⚠️  {key} sums to 0 — leaving values unchanged.")
        return lst

    factor = total / s
    normalized = [round(v * factor, 6) for v in lst]

    _print_list_diff(key, original, normalized, labels=labels, total=total)
    return normalized

def _normalize_dict_with_diffs(key: str, d: dict, total: float):
    """Normalize dict values to sum=total and print per-key diffs."""
    if not isinstance(d, dict):
        print(f"ℹ️  {key} is not a dict, skipping.")
        return d

    original = dict(d)

    # Coerce values to float where possible
    numeric = {}
    for k, v in d.items():
        try:
            numeric[k] = float(v)
        except Exception:
            print(f"⚠️  Skipping non-numeric value for {key}[{k!r}]: {v!r}")

    s = sum(numeric.values())
    if s == 0:
        print(f"⚠️  {key} sums to 0 — leaving values unchanged.")
        return d

    factor = total / s
    normalized = {k: round(v * factor, 6) for k, v in numeric.items()}

    _print_dict_diff(key, original, normalized, total=total)
    return normalized

def _print_list_diff(
    key: str,
    before: list,
    after: list,
    labels: list[str] | None = None,
    *,
    total: float = 100.0,
):
    print(f"\n🔄 Normalized {key} (sum {round(sum(before), 6)} → {total})")
    if labels and len(labels) == len(after):
        for name, b, a in zip(labels, before, after):
            if _changed(b, a):
                print(f"  • {name:>15}: {b}  →  {a}")
    else:
        for i, (b, a) in enumerate(zip(before, after)):
            if _changed(b, a):
                print(f"  • idx {i:>2}: {b}  →  {a}")
def _print_dict_diff(
    key: str,
    before: dict,
    after: dict,
    *,
    total: float = 100.0,
):
    print(
        f"\n🔄 Normalized {key} "
        f"(sum {round(sum(v for v in before.values() if isinstance(v, (int, float))), 6)} → {total})"
    )
    all_keys = set(before.keys()) | set(after.keys())

    # If the dict looks like colors → use WUBRG order
    if all(k in CANONICAL_COLOR_ORDER for k in all_keys):
        ordered = _ordered_color_keys(all_keys)
    else:
        ordered = sorted(all_keys)

    any_changed = False
    for k in ordered:
        b = before.get(k, 0.0)
        a = after.get(k, 0.0)
        if _changed(b, a):
            any_changed = True
            print(f"  • {k:>20}: {b}  →  {a}")
    if not any_changed:
        print("  • (no per-item changes)")

def _changed(a, b, eps: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) > eps
    except Exception:
        return a != b

def _derive_card_types(ctw: dict) -> list[str]:
    """
    Derive an ordered list of card types from a card_types_weights dict.
    Preference:
      1) Keys from the `_default` row (in that order), if it's a dict
      2) Then any keys from other dict rows, preserving first-seen order
    Non-dict rows are ignored.
    """
    if not isinstance(ctw, dict):
        return []
    derived: list[str] = []
    seen: set[str] = set()

    # 1) from _default dict, in order
    if isinstance(ctw.get("_default"), dict):
        for k in ctw["_default"].keys():
            if k not in seen:
                derived.append(k)
                seen.add(k)

    # 2) from other dict rows, in encounter order
    for color_key, row in ctw.items():
        if color_key == "_default" or not isinstance(row, dict):
            continue
        for k in row.keys():
            if k not in seen:
                derived.append(k)
                seen.add(k)

    return derived

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize and sanity-check config.yaml weights for readability")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument("--save", action="store_true", help="Overwrite the config file with normalized values")
    args = parser.parse_args()

    check_and_normalize_config(args.config_path, save=args.save)
