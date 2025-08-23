from __future__ import annotations

from scipy.stats import truncnorm

import numpy as np

import random

import yaml

import logging

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

def check_and_normalize_config(config_path: str, save: bool = False, total: float = 100.0, *, verbose: bool = True):
    """Enhanced normalization/validation with raw + merged phases and silent mode.

    Args:
        config_path: path to user config
        save: write normalized file
        total: target sum (usually 100)
        verbose: if False, suppress prints and just return normalized structure or None on error
    """
    path = Path(config_path)
    if not path.exists():
        if verbose:
            print(f"❌ Config file not found: {path}")
        return None
    defaults_path = path.parent / "DEFAULTSCONFIG.yml"
    if not defaults_path.exists():
        if verbose:
            print(f"❌ DEFAULTSCONFIG.yml not found at: {defaults_path}")
        return None

    # Load raw user (pre-merge) for raw-specific validation
    try:
        with open(path, 'r') as uf:
            raw_user = yaml.safe_load(uf) or {}
    except Exception:
        raw_user = {}

    import config_manager
    config = config_manager.load_config(str(path))  # merged
    with open(defaults_path, 'r') as df:
        defaults = yaml.safe_load(df) or {}

    # Raw validation (omission awareness)
    raw_issues = _validate_raw_user_config_structure(raw_user)
    if verbose:
        print("=" * 80)
        print("🔍 CONFIGURATION VALIDATION & NORMALIZATION")
        print("=" * 80)
        print("\n📋 Validating user configuration structure (raw before merge)...")
        print("-" * 40)
        _print_validation_results(raw_issues)

    # Merged validation (structure)
    if verbose:
        print("\n📋 Validating merged configuration structure...")
        print("-" * 40)
    merged_structure_issues = _validate_user_config_structure(config)
    stop_early = _print_validation_results(merged_structure_issues) if verbose else any("❌" in x for x in merged_structure_issues)
    if stop_early:
        if verbose:
            print("\n" + "=" * 80)
        return None

    # Integrity validation
    if verbose:
        print("\n📋 Validating merged configuration...")
        print("-" * 40)
    integrity_issues = _validate_config_integrity(config, defaults)
    stop = _print_validation_results(integrity_issues) if verbose else any("❌" in x for x in integrity_issues)
    if stop:
        if verbose:
            print("\n" + "=" * 80)
        return None

    if verbose:
        print("\n📊 WEIGHT NORMALIZATION")
        print("-" * 40)
    fixed = _normalize_all_weights_with_diffs(config, defaults, total=total, verbose=verbose)

    # Final validation
    if verbose:
        print("\n📋 Final validation after normalization...")
        print("-" * 40)
    final_issues = _validate_final_config(fixed)
    final_stop = _print_validation_results(final_issues) if verbose else any("❌" in x for x in final_issues)
    if final_stop:
        if verbose:
            print("\n" + "=" * 80)
        return None

    if save:
        with open(path, 'w') as outf:
            yaml.safe_dump(fixed, outf, sort_keys=False)
        if verbose:
            print("\n✅ CONFIGURATION SAVED")
            print("-" * 30)
            print(f"Normalized config saved to {path}")
    else:
        if verbose:
            print("\n💾 CONFIGURATION NOT SAVED")
            print("-" * 30)
            print("Normalization complete. Use --save to overwrite the file.")
    if verbose:
        print("=" * 80)
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

def _normalize_all_weights_with_diffs(config: dict, defaults: dict, total: float = 100.0, *, verbose: bool = True) -> dict:
    """
    Walk through config and normalize known weight sections:
      - skeleton_params.colors_weights   (dict preferred; list accepted)
      - skeleton_params.rarities_weights (dict preferred; list accepted)
      - skeleton_params.card_types_weights (dict: per-color rows; list accepted)
        including optional `_default` row
    Uses labels from:
      - skeleton_params.colors
      - skeleton_params.rarities (derived from rarities_weights keys)
      - skeleton_params.card_types
    Uses fallback values from defaults config.
    """

    total_cards = config["square_config"]["total_cards"]

    if config["pack_builder"]["enabled"]:
        countsum = 0
        for slot in config["pack_builder"]["pack"]:
            countsum += slot["count"]
        if countsum != total_cards:
            print(f"⚠️  WARNING: Updating total_cards from {total_cards} to {countsum} based on pack_builder counts.\n")
            total_cards = countsum
            config["square_config"]["total_cards"] = total_cards

    # Ensure skeleton_params exists and merge with defaults
    if "skeleton_params" not in config:
        config["skeleton_params"] = {}
    
    # Merge skeleton_params with defaults (user values take precedence)
    default_sp = defaults["skeleton_params"]
    sp = config["skeleton_params"]
    
    # Merge missing keys from defaults
    for key, value in default_sp.items():
        if key not in sp:
            sp[key] = value
    if not isinstance(sp, dict):
        print("⚠️  'skeleton_params' missing or not a dict; nothing to do.")
        return config

    colors = sp.get("colors", default_sp.get("colors", []))
    
    # Derive rarities from rarities_weights keys (eliminate redundancy)
    # After merge, rarities_weights should always be available from defaults
    rarities_weights = sp.get("rarities_weights", {})
    if isinstance(rarities_weights, dict):
        rarities = list(rarities_weights.keys())
    else:
        # This should not happen if DEFAULTSCONFIG.yml is properly structured
        print("⚠️  rarities_weights not found or not a dict - configuration may be incomplete")
        rarities = []
    
    # Always use canonical_card_types from defaults as the authoritative list
    # Don't derive from user's partial config as this loses missing types
    card_types = sp.get("card_types") or default_sp.get("canonical_card_types", [])
    
    # Update the user config to have the complete card_types list
    if card_types and card_types != sp.get("card_types"):
        sp["card_types"] = list(card_types)
        if verbose:
            print("ℹ️  CARD TYPES UPDATED")
            print("   " + "─" * 40)
            print("   Updated 'skeleton_params.card_types' to use complete canonical list.")

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
                if verbose:
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
                if verbose:
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

    # ---- NEW SCHEMA: card_types_color_defaults + user_overlays ----
    # Build per-color type weight maps from baselines (mode-specific) then apply overlays
    color_defaults_root = sp.get("card_types_color_defaults") or default_sp.get("card_types_color_defaults")
    if not color_defaults_root:
        print("❌ Missing 'card_types_color_defaults' in skeleton_params.")
        return config
    types_mode = sp.get("types_mode", "normal")
    if types_mode not in color_defaults_root:
        print(f"❌ Mode '{types_mode}' not found under card_types_color_defaults.")
        return config
    mode_defaults = color_defaults_root[types_mode]
    # Validate presence of all colors
    missing_colors = [c for c in colors if c not in mode_defaults]
    if missing_colors:
        print(f"❌ Mode '{types_mode}' missing color baselines: {missing_colors}")
        return config
    # New overlay mechanism: user supplies skeleton_params.card_types_color_weights with per-color partial overrides.
    user_color_overrides = sp.get("card_types_color_weights", {})
    if user_color_overrides and not isinstance(user_color_overrides, dict):
        print("⚠️  Ignoring non-dict card_types_color_weights (expected mapping of colors → type weights)")
        user_color_overrides = {}
    # Support special '_all' key for global type overrides across every color
    global_all_overrides = {}
    if "_all" in user_color_overrides:
        if isinstance(user_color_overrides["_all"], dict):
            global_all_overrides = user_color_overrides["_all"]
        else:
            print("⚠️  Ignoring non-dict _all in card_types_color_weights")
        # remove to avoid treating as color name
        user_color_overrides = {k: v for k, v in user_color_overrides.items() if k != "_all"}
    
    # Analyze and report user's configuration intent
    if verbose:
        print(f"\n🎯 USER CONFIGURATION ANALYSIS (mode: {types_mode})")
        print("─" * 60)
    
    if global_all_overrides:
        global_keys = sorted(global_all_overrides.keys())
        print(f"🌐 Global overrides (_all) detected for: {', '.join(global_keys)}")
        for k, v in sorted(global_all_overrides.items()):
            print(f"   • {k}: {v} (applies to ALL colors)")
    else:
        if verbose:
            print("🌐 No global overrides (_all) found")
    
    if user_color_overrides:
        print(f"🎨 Per-color overrides detected:")
        for color in sorted(user_color_overrides.keys()):
            if color in colors and isinstance(user_color_overrides[color], dict):
                override_keys = sorted(user_color_overrides[color].keys())
                print(f"   • {color}: {', '.join(override_keys)}")
                for k, v in sorted(user_color_overrides[color].items()):
                    print(f"     - {k}: {v}")
    else:
        if verbose:
            print("🎨 No per-color overrides found")
    
    if verbose:
        print(f"📊 Using baseline weights from mode '{types_mode}' and applying overlays...")
        print("─" * 60)
        
    # Prepare final structure similar to old card_types_weights
    final_weights: dict[str, dict[str, float]] = {}
    if verbose:
        print("\n📊 BUILDING TYPE WEIGHTS FROM BASELINES")
    for color in colors:
        baseline = dict(mode_defaults[color])
        # Normalize baseline if it does not sum to total
        b_sum = sum(float(v) for v in baseline.values())
        if b_sum <= 0:
            print(f"⚠️  Baseline for {color} sums to {b_sum}. Using uniform distribution.")
            baseline = {t: total / len(card_types) for t in card_types}
            b_sum = total
        # Scale baseline to total exactly
        scale = total / b_sum
        for k in list(baseline.keys()):
            if k not in card_types:
                if STRICT:
                    baseline.pop(k)
                else:
                    print(f"⚠️  {color} baseline has unknown type '{k}', keeping (STRICT off)")
        baseline = {k: float(baseline.get(k, 0.0)) * scale for k in card_types}
        row = dict(baseline)
        provenance_steps: list[str] = ["baseline"]
        applied_all_keys = []
        if global_all_overrides:
            for k, v in global_all_overrides.items():
                if k not in card_types:
                    if STRICT:
                        continue
                try:
                    row[k] = float(v)
                    applied_all_keys.append(k)
                except Exception:
                    print(f"⚠️  _all: non-numeric override {k}={v!r} skipped")
            if applied_all_keys:
                provenance_steps.append(f"_all({len(applied_all_keys)})")
        overrides = user_color_overrides.get(color, {}) if color in user_color_overrides else {}
        if overrides and not isinstance(overrides, dict):
            print(f"⚠️  Ignoring non-dict override for color {color}")
            overrides = {}
        # Apply overrides (absolute replacement of those type weights)
        applied_keys = []
        for k, v in overrides.items():
            if k not in card_types:
                if STRICT:
                    print(f"⚠️  {color}: dropping unknown type '{k}' in override")
                else:
                    print(f"⚠️  {color}: unknown type '{k}' kept (STRICT off)")
                continue
            try:
                row[k] = float(v)
                applied_keys.append(k)
            except Exception:
                print(f"⚠️  {color}: non-numeric override {k}={v!r} skipped")
        if applied_keys:
            provenance_steps.append(f"color({len(applied_keys)})")
        # Adjust remaining to keep total 100: proportionally scale non-overridden types
        new_sum = sum(row.values())
        if abs(new_sum - total) > 1e-6:
            # Identify adjustable pool (non-overridden)
            adjustable = [k for k in card_types if k not in applied_keys and k not in applied_all_keys]
            current_adjustable_sum = sum(row[k] for k in adjustable)
            if current_adjustable_sum <= 0:
                # Nothing adjustable; just normalize entire row
                row = _normalize_dict_with_diffs(
                    key=f"skeleton_params.card_types_color_weights[{color}]", d=row, total=total
                )
                final_weights[color] = row
                provenance_steps.append("normalize(all)")
                if verbose:
                    print(f"🔧 {color}: {' → '.join(provenance_steps)} → sum=100.0")
            else:
                # Compute factor so that overridden values remain fixed
                remaining_target = total - sum(row[k] for k in applied_keys + applied_all_keys)
                if remaining_target < 0:
                    # Overridden values alone exceed total -> normalize overridden + others
                    row = _normalize_dict_with_diffs(
                        key=f"skeleton_params.card_types_color_weights[{color}] (exceeded)", d=row, total=total
                    )
                    final_weights[color] = row
                    provenance_steps.append("normalize(exceeded)")
                    if verbose:
                        print(f"⚠️  {color}: {' → '.join(provenance_steps)} → sum=100.0 (exceeded)")
                else:
                    scale_factor = remaining_target / current_adjustable_sum if current_adjustable_sum > 0 else 0.0
                    for k in adjustable:
                        row[k] = row[k] * scale_factor
                    # Final rounding (no further normalization to preserve fixed overrides)
                    rounded = {k: round(v, 1) for k, v in row.items()}
                    adjust_sum = sum(rounded.values())
                    # Minor correction for rounding drift
                    drift = total - adjust_sum
                    if abs(drift) >= 0.1:
                        # Apply drift to largest adjustable bucket (first by order)
                        tgt = None
                        for k in adjustable:
                            if tgt is None or rounded[k] > rounded[tgt]:
                                tgt = k
                        if tgt:
                            rounded[tgt] = round(rounded[tgt] + drift, 1)
                    final_weights[color] = rounded
                    if scale_factor != 0:
                        provenance_steps.append(f"scale({scale_factor:.3f})")
                    exact_sum = sum(rounded.values())
                    if abs(exact_sum - total) > 1e-6:
                        residual = round(total - exact_sum, 10)
                        if adjustable:
                            largest = max(adjustable, key=lambda k: rounded[k])
                            rounded[largest] = round(rounded[largest] + residual, 1)
                    if verbose:
                        print(f"🔧 {color}: {' → '.join(provenance_steps)} → sum={sum(rounded.values()):.1f}")

        else:
            rounded = {k: round(v, 1) for k, v in row.items()}
            final_weights[color] = rounded
            r_sum = sum(rounded.values())
            if abs(r_sum - total) > 1e-6:
                residual = round(total - r_sum, 10)
                largest_key = max(rounded, key=lambda k: rounded[k])
                rounded[largest_key] = round(rounded[largest_key] + residual, 1)
            if verbose:
                print(f"🔧 {color}: {' → '.join(provenance_steps)} (unchanged) → sum={sum(rounded.values()):.1f}")

    # Store back in legacy key for runtime compatibility
    sp["card_types_weights"] = final_weights

    # Display final type weights as a pretty table
    if verbose:
        _print_type_weights_table(final_weights, card_types)

    return config


# ---------- helpers ----------

def _print_type_weights_table(final_weights: dict, card_types: list[str]) -> None:
    """Print a pretty table showing final type weights with colors as rows and types as columns."""
    if not final_weights or not card_types:
        return
    
    print("\n📊 FINAL TYPE WEIGHTS TABLE")
    print("─" * 120)
    
    # Get all colors in consistent order
    colors = list(final_weights.keys())
    if CANONICAL_COLOR_ORDER:
        # Sort colors by canonical order if available
        color_order = []
        for canonical_color in CANONICAL_COLOR_ORDER:
            if canonical_color in colors:
                color_order.append(canonical_color)
        # Add any remaining colors not in canonical order
        for color in colors:
            if color not in color_order:
                color_order.append(color)
        colors = color_order
    
    # Calculate column widths
    color_col_width = max(len(c) for c in colors) + 1 if colors else 8
    type_col_width = 9  # Width for weight values
    
    # Print header with types as columns
    header = f"{'COLOR':<{color_col_width}}"
    for card_type in card_types:
        header += f"{card_type[:8]:>{type_col_width}}"  # Truncate long type names
    header += f"{'TOTAL':>{type_col_width}}"
    print(header)
    print("─" * len(header))
    
    # Print each color row
    for color in colors:
        row = f"{color:<{color_col_width}}"
        color_total = 0.0
        for card_type in card_types:
            weight = final_weights[color].get(card_type, 0.0)
            row += f"{weight:>{type_col_width-1}.1f} "
            color_total += weight
        
        # Add total and warning if not exactly 100
        total_str = f"{color_total:.1f}"
        if abs(color_total - 100.0) > 0.1:
            total_str += "⚠️"
        row += f"{total_str:>{type_col_width}}"
        print(row)
    
    print("─" * len(header))
    print("\n� This table shows the final type distribution that will be used for card generation.")
    print("   Colors with totals ≠ 100.0 are marked with ⚠️")


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
    print(f"\n� NORMALIZING: {key}")
    print("   " + "─" * 50)
    print(f"   Sum: {round(sum(before), 6)} → {total}")
    if labels and len(labels) == len(after):
        for name, b, a in zip(labels, before, after):
            if _changed(b, a):
                print(f"   • {name:>15}: {b:>6.1f}  →  {a:>6.1f}")
    else:
        for i, (b, a) in enumerate(zip(before, after)):
            if _changed(b, a):
                print(f"   • idx {i:>2}: {b:>6.1f}  →  {a:>6.1f}")
                
def _print_dict_diff(
    key: str,
    before: dict,
    after: dict,
    *,
    total: float = 100.0,
):
    before_sum = round(sum(v for v in before.values() if isinstance(v, (int, float))), 6)
    print(f"\n📏 NORMALIZING: {key}")
    print("   " + "─" * 50)
    print(f"   Sum: {before_sum} → {total}")
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
            print(f"   • {k:>20}: {b:>6.1f}  →  {a:>6.1f}")
    if not any_changed:
        print("   • (no per-item changes)")

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
    default_values: dict = None,
    total: float = 100.0,
):
    """Print detailed breakdown showing: User Set → Defaults Applied → Final Value."""
    print(f"\n🔄 SMART PARTIAL LOGIC: {key}")
    print("   " + "─" * 60)
    print(f"   Target sum: {total:.1f}")
    
    all_keys = set(user_values.keys()) | set(final_values.keys())
    if default_values:
        all_keys |= set(default_values.keys())
    
    # If the dict looks like colors → use WUBRG order
    if all(k in CANONICAL_COLOR_ORDER for k in all_keys):
        ordered = _ordered_color_keys(all_keys)
    else:
        ordered = sorted(all_keys)
    
    preserved_keys = []
    filled_keys = []
    
    for k in ordered:
        user_val = user_values.get(k, None)
        default_val = default_values.get(k, 0.0) if default_values else 0.0
        final_val = final_values.get(k, 0.0)
        
        if final_val == 0:  # Skip zero values to reduce noise
            continue
            
        if user_val is not None and user_val != 0:
            preserved_keys.append((k, final_val))
            if default_values:
                print(f"   • {k:>20}: {user_val:>6.1f} (user) → {'-':>8} (unused) → {final_val:>6.1f} (preserved)")
            else:
                print(f"   • {k:>20}: {final_val:<8.1f} (preserved)")
        elif final_val > 0:
            filled_keys.append((k, final_val))
            if default_values:
                print(f"   • {k:>20}: {'-':>6} (user) → {default_val:>6.1f} (default) → {final_val:>6.1f} (filled)")
            else:
                print(f"   • {k:>20}: {final_val:<8.1f} (filled)")
    
    # Show zero values user explicitly set
    for k in ordered:
        user_val = user_values.get(k, None)
        final_val = final_values.get(k, 0.0)
        
        if user_val == 0 and k in user_values:  # User explicitly set to 0
            if default_values:
                default_val = default_values.get(k, 0.0)
                print(f"   • {k:>20}: {user_val:>6.1f} (user) → {'-':>8} (overridden) → {final_val:>6.1f} (zeroed)")
            else:
                print(f"   • {k:>20}: {final_val:<8.1f} (user set to 0)")
    
    # Summary
    preserved_sum = sum(val for _, val in preserved_keys)
    filled_sum = sum(val for _, val in filled_keys)
    print(f"   " + "─" * 60)
    print(f"   📊 Summary: {preserved_sum:.1f} preserved + {filled_sum:.1f} filled = {total:.1f}")

def _print_types_mode_overlay(types_mode: str, base_defaults: dict, profile_weights: dict, final_weights: dict, total: float = 100.0):
    """
    Display the types_mode profile overlay process showing how base defaults are overlaid with profile weights.
    """
    print(f"\n🔄 TYPES_MODE PROFILE OVERLAY: {types_mode}")
    print("   " + "─" * 64)
    print(f"   Base (_default) → Profile (_{types_mode}Defaults) → Result")
    
    # Get all types from any of the dicts
    all_types = set(base_defaults.keys()) | set(profile_weights.keys()) | set(final_weights.keys())
    
    for card_type in sorted(all_types):
        base_val = base_defaults.get(card_type, 0.0)
        profile_val = profile_weights.get(card_type, None)
        final_val = final_weights.get(card_type, 0.0)
        
        if profile_val is not None:
            # Profile provided this type
            print(f"   • {card_type:>18}: {base_val:>6.1f} (base) → {profile_val:>6.1f} (profile) → {final_val:>6.1f} (result)")
        else:
            # Type filled from base with scaling
            if base_val > 0 and final_val != base_val:
                print(f"   • {card_type:>18}: {base_val:>6.1f} (base) →      - (scaled)  → {final_val:>6.1f} (result)")
            elif base_val > 0:
                print(f"   • {card_type:>18}: {base_val:>6.1f} (base) →      - (kept)    → {final_val:>6.1f} (result)")
            else:
                print(f"   • {card_type:>18}: {base_val:>6.1f} (base) →      - (unused)  → {final_val:>6.1f} (result)")
    
    print("   " + "─" * 64)
    base_sum = sum(base_defaults.values())
    profile_sum = sum(v for v in profile_weights.values() if v is not None)
    final_sum = sum(final_weights.values())
    scaling_sum = final_sum - profile_sum
    print(f"   📊 Summary: {base_sum:.1f} (base) → {profile_sum:.1f} (profile) + {scaling_sum:.1f} (scaled) = {final_sum:.1f}")

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

def _validate_final_config(config: dict) -> list[str]:
    """
    Final validation after normalization - catch issues that would break the actual system.
    """
    issues = []
    
    if "skeleton_params" not in config:
        return issues
    
    sp = config["skeleton_params"]
    
    # Check for completely zero final weights after all processing (this would break generation)
    if "card_types_weights" in sp:
        ctw = sp["card_types_weights"]
        if "_default" in ctw:
            default_weights = ctw["_default"]
            if isinstance(default_weights, dict):
                total_weight = sum(v for v in default_weights.values() if isinstance(v, (int, float)) and v > 0)
                if total_weight == 0:
                    issues.append("❌ ERROR: Final _default card types sum to 0 - no cards can be generated. Check your configuration.")
    
    # Check for missing color weights
    if "colors_weights" not in sp:
        issues.append("❌ ERROR: Missing 'colors_weights' - cannot determine which colors to use")
    elif isinstance(sp["colors_weights"], dict):
        color_total = sum(v for v in sp["colors_weights"].values() if isinstance(v, (int, float)) and v > 0)
        if color_total == 0:
            issues.append("❌ ERROR: All color weights are 0 - no colors can be selected")
    
    # Check for missing rarity weights  
    if "rarities_weights" not in sp:
        issues.append("❌ ERROR: Missing 'rarities_weights' - cannot determine card rarities")
    elif isinstance(sp["rarities_weights"], dict):
        rarity_total = sum(v for v in sp["rarities_weights"].values() if isinstance(v, (int, float)) and v > 0)
        if rarity_total == 0:
            issues.append("❌ ERROR: All rarity weights are 0 - no rarities can be selected")
    
    return issues


def _validate_user_config_structure(config: dict) -> list[str]:
    """
    Validate user configuration structure for critical issues before merging with defaults.
    This catches problems that would be masked by the defaults.
    """
    issues = []
    
    # Check if skeleton_params exists at root level (correct structure)
    if "skeleton_params" not in config:
        issues.append("⚠️  WARNING: Missing 'skeleton_params' section - will use all defaults")
        return issues
    
    skeleton_params = config["skeleton_params"]
    if not isinstance(skeleton_params, dict):
        issues.append("❌ ERROR: 'skeleton_params' must be a dictionary")
        return issues
    
    # Check skeleton_params structure (at root level)
    sp = skeleton_params
    if not isinstance(sp, dict):
        issues.append("❌ ERROR: 'skeleton_params' must be a dictionary")
    else:
        # Check types_mode validity - but only for basic structure, not specific modes
        if "types_mode" in sp:
            types_mode = sp["types_mode"]
            if not isinstance(types_mode, str):
                issues.append("❌ ERROR: 'types_mode' must be a string")
            elif types_mode == "":
                issues.append("❌ ERROR: 'types_mode' cannot be empty - use 'normal' for default behavior")
    
    # Check card_types_weights structure
    if "card_types_weights" in skeleton_params:
        ctw = skeleton_params["card_types_weights"]
        if not isinstance(ctw, dict):
            issues.append("❌ ERROR: 'card_types_weights' must be a dictionary")
        else:
            # Check for negative weights only - don't validate zero sums yet
            if "_default" in ctw:
                default_weights = ctw["_default"]
                if isinstance(default_weights, dict):
                    # Only check for negative weights - zero sums will be handled after merge
                    for card_type, weight in default_weights.items():
                        if isinstance(weight, (int, float)) and weight < 0:
                            issues.append(f"❌ ERROR: Negative weight in _default.{card_type}: {weight}")
                
                elif not isinstance(default_weights, dict):
                    issues.append("❌ ERROR: card_types_weights._default must be a dictionary")
            
            # Check profiles for structural issues
            for key, weights in ctw.items():
                if key.startswith('_') and key.endswith('Defaults'):
                    if not isinstance(weights, dict):
                        issues.append(f"❌ ERROR: Profile '{key}' must be a dictionary of card type weights")
                    elif isinstance(weights, dict):
                        # Check for negative weights in profiles
                        for card_type, weight in weights.items():
                            if isinstance(weight, (int, float)) and weight < 0:
                                issues.append(f"❌ ERROR: Negative weight in {key}.{card_type}: {weight}")
    
    return issues


def _validate_raw_user_config_structure(raw: dict) -> list[str]:
    """Pre-merge raw user config validation (to detect omissions)."""
    issues: list[str] = []
    if not raw:
        issues.append("ℹ️  INFO: Empty user config (defaults only)")
        return issues
    sp = raw.get("skeleton_params")
    if sp is None:
        issues.append("⚠️  WARNING: Missing 'skeleton_params' in user config (defaults will supply it)")
        return issues
    if not isinstance(sp, dict):
        issues.append("❌ ERROR: 'skeleton_params' must be a dict in user config")
        return issues
    if 'types_mode' not in sp:
        issues.append("ℹ️  INFO: 'types_mode' not specified; default 'normal' assumed")
    return issues


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
    default_sp = defaults["skeleton_params"]
    
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
    
    # 2. Check for missing critical sections / deprecated schemas
    # colors_weights still required
    if "colors_weights" not in sp:
        issues.append("⚠️  WARNING: Missing 'colors_weights' section, will use defaults")

    # Legacy vs new type weight schema handling:
    # New schema: card_types_color_defaults (in defaults) + optional card_types_color_weights (user overrides)
    # Legacy schema: card_types_weights provided directly by user
    has_legacy_types = "card_types_weights" in sp
    # Detect explicit user overrides in new schema
    has_new_overrides = "card_types_color_weights" in sp

    if has_legacy_types:
        # Deprecation warning only if user supplied (defaults no longer include this key pre-normalization)
        issues.append("⚠️  WARNING: Detected deprecated 'card_types_weights' schema; please migrate to 'card_types_color_defaults' + 'card_types_color_weights'.")
    elif not has_new_overrides:
        # Neither legacy nor new override layer supplied → using pure mode baselines
        issues.append("⚠️  WARNING: No type weight overrides supplied (neither legacy 'card_types_weights' nor new 'card_types_color_weights'); using mode baseline defaults.")
    
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
            
            # Check for structural issues that break the three-layer system
            for key, weights in ctw.items():
                if key.startswith('_') and key.endswith('Defaults'):
                    # This is a profile (like _squareDefaults)
                    if not isinstance(weights, dict):
                        issues.append(f"❌ ERROR: Profile '{key}' must be a dictionary of card type weights")
                    elif len(weights) == 0:
                        issues.append(f"⚠️  WARNING: Profile '{key}' is empty - will have no effect")
            
            # Validate individual card type weights
            for key, weights in ctw.items():
                if isinstance(weights, dict):
                    total_weight = sum(v for v in weights.values() if isinstance(v, (int, float)))
                    
                    # Check for extremely low individual type weights
                    for card_type, weight in weights.items():
                        if isinstance(weight, (int, float)):
                            if weight < 0:
                                issues.append(f"❌ ERROR: Negative weight in {key}.{card_type}: {weight}")
                            elif weight > 200:  # Suspiciously high
                                issues.append(f"⚠️  WARNING: Very high weight in {key}.{card_type}: {weight}")
                    
                    # Only flag zero-sum if ALL provided weights are zero (suspicious)
                    if len(weights) > 5:  # User provided many types
                        non_zero_count = sum(1 for v in weights.values() if isinstance(v, (int, float)) and v > 0)
                        if total_weight == 0 and non_zero_count == 0:
                            if key in CANONICAL_COLOR_ORDER:
                                # Zero weights for a color that could be selected as primary
                                issues.append(f"⚠️  WARNING: card_types_weights[{key}] has all zero weights - this color cannot generate cards if selected as primary color. Consider: (1) removing {key} from colors_weights, (2) setting non-zero type weights, or (3) using {key} only for color bleed.")
                            else:
                                # Zero weights for _default or other section is more critical only if user provided extensive config
                                issues.append(f"❌ ERROR: card_types_weights[{key}] has all zero weights - this will cause generation failures")
    
    # 4. Check for cross-reference issues: colors with weights but no viable types
    if "colors_weights" in sp and "card_types_weights" in sp:
        color_weights = sp["colors_weights"]
        type_weights = sp["card_types_weights"]
        
        if isinstance(color_weights, dict) and isinstance(type_weights, dict):
            for color, color_weight in color_weights.items():
                if isinstance(color_weight, (int, float)) and color_weight > 0:
                    # This color can be selected as primary color
                    if color in type_weights:
                        color_types = type_weights[color]
                        if isinstance(color_types, dict):
                            total_type_weight = sum(v for v in color_types.values() if isinstance(v, (int, float)))
                            if total_type_weight == 0:
                                issues.append(f"⚠️  WARNING: Color '{color}' has {color_weight}% selection weight but 0% type weights - cards will have type='None' if this color is selected")
    
    # 5. Check for deprecated or suspicious keys
    suspicious_keys = ["card_type", "color", "rarity"]  # Common typos
    for key in sp.keys():
        if key in suspicious_keys:
            issues.append(f"⚠️  WARNING: Suspicious key '{key}' - did you mean '{key}s_weights'?")

    # 7. Check for potential profile conflicts
    if "card_types_weights" in sp:
        ctw = sp["card_types_weights"]
        types_mode = sp.get("types_mode", "normal")
        
        # Warn about potential conflicts between user overrides and profile logic
        if types_mode != "normal" and "_default" in ctw:
            user_default = ctw["_default"]
            if isinstance(user_default, dict) and len(user_default) > 8:
                # User provided extensive _default override - might conflict with profile
                issues.append(f"ℹ️  INFO: Extensive _default override with types_mode '{types_mode}'. The profile will be applied after your _default, which may override some of your settings. Consider using a color-specific override instead.")
    
    # 8. Check generation parameters - use correct config parameter names
    square_config = config.get("square_config", {})
    if "total_cards" in square_config:
        total_cards = square_config["total_cards"]
        if isinstance(total_cards, (int, float)):
            if total_cards <= 0:
                issues.append("❌ ERROR: 'total_cards' must be positive")
            elif total_cards > 1000:
                issues.append("⚠️  WARNING: Very high card count may take a long time to generate")
        if config["pack_builder"]["enabled"]:
            countsum = 0
            for slot in config["pack_builder"]["pack"]:
                countsum += slot["count"]
            if countsum != total_cards:
                issues.append(f"⚠️  WARNING: Pack builder counts sum ({countsum}) does not match total_cards ({total_cards}) Using pack_builder countsum as total_cards.")

    # 9. Check concurrency parameters
    if "concurrency" in square_config:
        concurrency = square_config["concurrency"]
        if isinstance(concurrency, (int, float)):
            if concurrency <= 0:
                issues.append("❌ ERROR: 'concurrency' must be positive")
            elif concurrency > 20:
                issues.append("⚠️  WARNING: Very high concurrency may cause performance issues")

    return issues

def _print_validation_results(issues: list[str]):
    """
    Print validation issues and determine if execution should stop.
    Returns True if there are critical errors that should stop execution.
    """
    if not issues:
        print("✅ No issues found - configuration is valid!")
        return False
    
    errors = [issue for issue in issues if "❌ ERROR:" in issue]
    warnings = [issue for issue in issues if "⚠️  WARNING:" in issue]
    info_messages = [issue for issue in issues if "ℹ️  INFO:" in issue]
    
    if errors:
        print(f"\n❌ CRITICAL ERRORS DETECTED ({len(errors)}):")
        print("   " + "─" * 60)
        for error in errors:
            print(f"   {error}")
        
        print(f"\n🛠️  HOW TO FIX CRITICAL ERRORS:")
        print("   " + "─" * 30)
        
        for error in errors:
            if "types_mode" in error and "references missing profile" in error:
                print("   • Missing Profile Error:")
                print("     - Check your types_mode value in skeleton_params")
                print("     - For 'square' mode, ensure '_squareDefaults' exists in card_types_weights")
                print("     - For custom modes, create '_<mode>Defaults' profile")
                print()
            elif "Invalid types_mode" in error:
                print("   • Invalid types_mode:")
                print("     - Use 'normal' for standard play-ready generation")
                print("     - Use 'square' for cube-optimized generation")
                print("     - Check for typos in your types_mode value")
                print()
            elif "sums to 0" in error and "_default" in error:
                print("   • Zero-sum Card Types:")
                print("     - _default profile cannot have all zero weights")
                print("     - Provide at least some positive card type weights")
                print("     - Partial overrides are OK - missing types will be filled automatically")
                print()
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        print("   " + "─" * 50)
        for warning in warnings:
            print(f"   {warning}")
    
    if info_messages:
        print(f"\nℹ️  INFORMATION ({len(info_messages)}):")
        print("   " + "─" * 50)
        for info in info_messages:
            print(f"   {info}")
    
    # Summary and decision
    if errors:
        print(f"\n🚨 VALIDATION FAILED: {len(errors)} critical error(s) must be fixed before proceeding!")
        print("   Configuration processing has been STOPPED.")
        return True
    elif warnings:
        print(f"\n💡 VALIDATION PASSED: {len(warnings)} warning(s) found - continuing with processing.")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED: Configuration is clean!")
        return False

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize and sanity-check config.yaml weights for readability")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument("--save", action="store_true", help="Overwrite the config file with normalized values")
    args = parser.parse_args()

    check_and_normalize_config(args.config_path, save=args.save)
