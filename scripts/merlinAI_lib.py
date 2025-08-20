from __future__ import annotations

from scipy.stats import truncnorm

import numpy as np

import random

import yaml

from pathlib import Path

_EPS = 1e-12


# NOTE: These constants have been moved to configs/DEFAULTSCONFIG.yml
# Import here for backward compatibility with config checker
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
    assert 0 <= mutation_chance <= 100, "mutation_chance must be between 0 and 100"

    d100 = random.random() * 100

    return d100 <= mutation_chance

# ========= CLI: Check & Normalize Config =========

def check_and_normalize_config(config_path: str, save: bool = False, total: float = 100.0):
    """
    User-facing CLI helper to check + normalize weights in a config.yaml.

    - Loads DEFAULTSCONFIG.yml first for fallback values
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

    # Load defaults first
    defaults_path = path.parent / "DEFAULTSCONFIG.yml"
    if not defaults_path.exists():
        print(f"❌ DEFAULTSCONFIG.yml not found at: {defaults_path}")
        return
        
    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f) or {}

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Perform validation checks before normalization
    print("🔍 Validating configuration...")
    validation_issues = _validate_config_integrity(config, defaults)
    _print_validation_results(validation_issues)
    
    # Check if there are critical errors that should stop processing
    critical_errors = [issue for issue in validation_issues if "❌ ERROR:" in issue]
    if critical_errors:
        print("\n🛑 Stopping due to critical errors. Please fix the errors above and try again.")
        return None

    print("\n📊 Normalizing weights...")
    fixed = _normalize_all_weights_with_diffs(config, defaults, total=total)

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

def _normalize_all_weights_with_diffs(config: dict, defaults: dict, total: float = 100.0) -> dict:
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
    Uses fallback values from defaults config.
    """
    sp = config.get("skeleton_params", {})
    if not isinstance(sp, dict):
        print("⚠️  'skeleton_params' missing or not a dict; nothing to do.")
        return config

    # Get defaults for skeleton_params
    default_sp = defaults.get("skeleton_params", {})
    
    colors = sp.get("colors", default_sp.get("colors", []))
    rarities = sp.get("rarities", default_sp.get("rarities", []))
    
    # Always use canonical_card_types from defaults as the authoritative list
    # Don't derive from user's partial config as this loses missing types
    card_types = sp.get("card_types") or default_sp.get("canonical_card_types", [])
    
    # Update the user config to have the complete card_types list
    if card_types and card_types != sp.get("card_types"):
        sp["card_types"] = list(card_types)
        print("ℹ️  Updated 'skeleton_params.card_types' to use complete canonical list.")

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
        
        # Get default type weights from defaults config (now in card_types_weights._default)
        default_sp_ctw = default_sp.get("card_types_weights", {})
        default_type_weights = default_sp_ctw.get("_default", {})

        # 1) Build the baseline default_map with smart filling logic
        #    - If user provided ALL types: normalize their input directly
        #    - If user provided PARTIAL types: preserve user values, scale missing ones to fit remainder
        if isinstance(ctw.get("_default"), list):
            user_default_raw = _list_to_labeled_dict(ctw["_default"], card_types)
        elif isinstance(ctw.get("_default"), dict):
            user_default_raw = dict(ctw["_default"])
        else:
            user_default_raw = {}

        # Coerce user values to numeric
        user_default_numeric = {}
        for t in card_types:
            if t in user_default_raw:
                try:
                    user_default_numeric[t] = float(user_default_raw[t])
                except Exception:
                    print(f"⚠️  Skipping non-numeric user value for type {t!r}: {user_default_raw[t]!r}")

        # Check if user provided all types or just partial
        user_provided_types = set(user_default_numeric.keys())
        all_types = set(card_types)
        missing_types = all_types - user_provided_types
        
        if missing_types:
            # PARTIAL input: preserve user values, scale missing ones to fit remainder
            user_sum = sum(user_default_numeric.values())
            remaining = total - user_sum
            
            if remaining < 0:
                print(f"⚠️  User _default values sum to {user_sum} > {total}, will normalize all values")
                # User exceeded total, normalize everything
                default_map_coerced = user_default_numeric
                for t in missing_types:
                    default_map_coerced[t] = 0.0
            else:
                # Fill missing types proportionally from defaults to use remaining percentage
                missing_defaults = {t: default_type_weights.get(t, 0.0) for t in missing_types}
                missing_sum = sum(missing_defaults.values())
                
                default_map_coerced = dict(user_default_numeric)  # Preserve user values exactly
                
                if missing_sum > 0 and remaining > 0:
                    # Scale missing types to fit in remaining space
                    scale_factor = remaining / missing_sum
                    for t in missing_types:
                        default_map_coerced[t] = missing_defaults[t] * scale_factor
                else:
                    # No defaults or no remaining space, set missing to 0
                    for t in missing_types:
                        default_map_coerced[t] = 0.0
                        
                print(f"ℹ️  Preserved user _default values, filled {len(missing_types)} missing types with {remaining:.1f}% remaining")
        else:
            # COMPLETE input: user provided all types, normalize their input
            default_map_coerced = user_default_numeric
            print(f"ℹ️  User provided complete _default, normalizing all values")

        # Only normalize if user exceeded total or provided complete input
        if missing_types and sum(default_map_coerced.values()) <= total + 0.001:  # Small tolerance for floating point
            # Don't normalize - we already have the right total
            default_norm = {k: round(v, 1) for k, v in default_map_coerced.items()}
            current_sum = sum(default_norm.values())
            _print_smart_partial_result(
                key="skeleton_params.card_types_weights[_default]",
                user_values=user_default_raw,
                final_values=default_norm,
                total=current_sum
            )
        else:
            # Normalize (either complete user input or user exceeded total)
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

            # Apply same smart partial logic as _default
            user_provided_types = set(row_num.keys())
            all_types = set(card_types)
            missing_types = all_types - user_provided_types
            
            if missing_types:
                # PARTIAL input: preserve user values, scale missing ones to fit remainder
                user_sum = sum(row_num.values())
                remaining = total - user_sum
                
                if remaining < 0:
                    print(f"⚠️  User {color_key} values sum to {user_sum} > {total}, will normalize all values")
                    # User exceeded total, fill missing from default and normalize everything
                    merged = dict(default_norm)
                    for t in row_num.keys() & set(card_types):
                        merged[t] = row_num[t]
                    # Will normalize below
                    final_row = merged
                    normalize_needed = True
                else:
                    # Fill missing types proportionally from default to use remaining percentage
                    final_row = dict(row_num)  # Preserve user values exactly
                    
                    if remaining > 0:
                        # Scale missing types from default to fit in remaining space
                        missing_defaults = {t: default_norm[t] for t in missing_types}
                        missing_sum = sum(missing_defaults.values())
                        
                        if missing_sum > 0:
                            scale_factor = remaining / missing_sum
                            for t in missing_types:
                                final_row[t] = missing_defaults[t] * scale_factor
                        else:
                            # No defaults for missing types, set to 0
                            for t in missing_types:
                                final_row[t] = 0.0
                    else:
                        # No remaining space, set missing to 0
                        for t in missing_types:
                            final_row[t] = 0.0
                            
                    print(f"ℹ️  Preserved user {color_key} values, filled {len(missing_types)} missing types with {remaining:.1f}% remaining")
                    normalize_needed = False
            else:
                # COMPLETE input: user provided all types, merge over default and normalize
                merged = dict(default_norm)
                for t in row_num.keys() & set(card_types):
                    merged[t] = row_num[t]
                final_row = merged
                normalize_needed = True
                print(f"ℹ️  User provided complete {color_key}, normalizing all values")

            # Only normalize if needed
            if normalize_needed:
                ctw[color_key] = _normalize_dict_with_diffs(
                    key=f"skeleton_params.card_types_weights[{color_key}] (resolved over _default)",
                    d=final_row,
                    total=total,
                )
            else:
                # Don't normalize - we already have the right total
                final_rounded = {k: round(v, 1) for k, v in final_row.items()}
                current_sum = sum(final_rounded.values())
                _print_smart_partial_result(
                    key=f"skeleton_params.card_types_weights[{color_key}]",
                    user_values=row_map,
                    final_values=final_rounded,
                    total=current_sum
                )
                ctw[color_key] = final_rounded

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
    normalized = {k: round(v * factor, 1) for k, v in numeric.items()}

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

def _print_smart_partial_result(
    key: str,
    user_values: dict,
    final_values: dict,
    *,
    total: float = 100.0,
):
    """Print detailed breakdown of smart partial logic showing preserved vs filled values."""
    print(f"\n✨ Smart partial logic applied to {key} (sum {total})")
    
    all_keys = set(user_values.keys()) | set(final_values.keys())
    
    # If the dict looks like colors → use WUBRG order
    if all(k in CANONICAL_COLOR_ORDER for k in all_keys):
        ordered = _ordered_color_keys(all_keys)
    else:
        ordered = sorted(all_keys)
    
    preserved_keys = []
    filled_keys = []
    
    for k in ordered:
        user_val = user_values.get(k, 0.0)
        final_val = final_values.get(k, 0.0)
        
        if k in user_values and user_val > 0:
            preserved_keys.append((k, final_val))
            print(f"  • {k:>20}: {final_val:<8.1f} (preserved)")
        elif final_val > 0:
            filled_keys.append((k, final_val))
            print(f"  • {k:>20}: {final_val:<8.1f} (filled)")
    
    # Summary
    preserved_sum = sum(val for _, val in preserved_keys)
    filled_sum = sum(val for _, val in filled_keys)
    print(f"  Summary: {preserved_sum:.1f} preserved + {filled_sum:.1f} filled = {total}")

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

def _validate_config_integrity(config: dict, defaults: dict) -> list[str]:
    """
    Perform additional validation checks on the configuration.
    Returns a list of warning/error messages.
    """
    issues = []
    
    # Check if skeleton_params exists
    if "skeleton_params" not in config:
        issues.append("❌ CRITICAL: Missing 'skeleton_params' section")
        return issues
    
    sp = config["skeleton_params"]
    default_sp = defaults.get("skeleton_params", {})
    
    # 1. Check for extremely unbalanced color weights
    if "colors_weights" in sp:
        colors = sp["colors_weights"]
        if isinstance(colors, dict):
            total = sum(v for v in colors.values() if isinstance(v, (int, float)))
            if total > 0:
                max_weight = max(v for v in colors.values() if isinstance(v, (int, float)))
                min_weight = min(v for v in colors.values() if isinstance(v, (int, float)) and v > 0)
                if max_weight / min_weight > 20:  # More than 20:1 ratio
                    issues.append(f"⚠️  WARNING: Extremely unbalanced color weights (max {max_weight}, min {min_weight})")
    
    # 2. Check for missing critical sections
    critical_sections = ["colors_weights", "card_types_weights"]
    for section in critical_sections:
        if section not in sp:
            issues.append(f"⚠️  WARNING: Missing '{section}' section, will use defaults")
    
    # 3. Check card_types_weights structure
    if "card_types_weights" in sp:
        ctw = sp["card_types_weights"]
        if isinstance(ctw, dict):
            # Check if _default exists
            if "_default" not in ctw:
                issues.append("⚠️  WARNING: Missing '_default' in card_types_weights, system may behave unexpectedly")
            
            # Check for color-specific weights without _default
            color_keys = [k for k in ctw.keys() if k in CANONICAL_COLOR_ORDER]
            if color_keys and "_default" not in ctw:
                issues.append("⚠️  WARNING: Color-specific weights defined without '_default' baseline")
            
            # Check for extremely low or high individual type weights
            for key, weights in ctw.items():
                if isinstance(weights, dict):
                    for card_type, weight in weights.items():
                        if isinstance(weight, (int, float)):
                            if weight < 0:
                                issues.append(f"❌ ERROR: Negative weight in {key}.{card_type}: {weight}")
                            elif weight > 200:  # Suspiciously high
                                issues.append(f"⚠️  WARNING: Very high weight in {key}.{card_type}: {weight}")
    
    # 4. Check for deprecated or suspicious keys
    suspicious_keys = ["card_type", "color", "rarity"]  # Common typos
    for key in sp.keys():
        if key in suspicious_keys:
            issues.append(f"⚠️  WARNING: Suspicious key '{key}' - did you mean '{key}s_weights'?")
    
    # 5. Check generation parameters
    if "generations" in sp:
        gen = sp["generations"]
        if isinstance(gen, (int, float)):
            if gen <= 0:
                issues.append("❌ ERROR: 'generations' must be positive")
            elif gen > 1000:
                issues.append("⚠️  WARNING: Very high generation count may take a long time")
    
    # 6. Check thread safety
    if "threads" in sp:
        threads = sp["threads"]
        if isinstance(threads, (int, float)):
            if threads <= 0:
                issues.append("❌ ERROR: 'threads' must be positive")
            elif threads > 20:
                issues.append("⚠️  WARNING: Very high thread count may cause performance issues")
    
    return issues

def _print_validation_results(issues: list[str]):
    """Print validation issues in a organized way."""
    if not issues:
        print("✅ Configuration validation passed - no issues found!")
        return
    
    errors = [issue for issue in issues if "❌ ERROR:" in issue]
    warnings = [issue for issue in issues if "⚠️  WARNING:" in issue]
    
    if errors:
        print(f"\n❌ ERRORS FOUND ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  {warning}")
    
    if errors:
        print("\n🚨 Configuration has ERRORS that should be fixed before use!")
    elif warnings:
        print("\n💡 Configuration has warnings but should work correctly.")

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize and sanity-check config.yaml weights for readability")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument("--save", action="store_true", help="Overwrite the config file with normalized values")
    args = parser.parse_args()

    check_and_normalize_config(args.config_path, save=args.save)
