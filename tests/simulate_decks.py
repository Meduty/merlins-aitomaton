#!/usr/bin/env python3
"""Deck type distribution simulation using real SkeletonParams + card_skeleton_generator.

Generates N synthetic "decks" by invoking the same machinery the generator
uses for per-card type/color selection, without calling any external APIs.

Counts card types (each individual primary type token in the skeleton 'type' field;
if a skeleton mutates into multiple types separated by commas, each counts once).

Usage:
  python scripts/simulate_decks.py --config configs/merlinSquare01.yml --modes normal play --decks 100 --deck-size 60
"""
from __future__ import annotations
import argparse, random, sys, os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any

# Ensure scripts package importable when running standalone
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.append(str(ROOT / 'scripts'))

from merlinAI_lib import check_and_normalize_config, check_mutation  # type: ignore
from square_generator import SkeletonParams, APIParams, card_skeleton_generator  # type: ignore
import config_manager  # type: ignore

PRIMARY_TYPES = [
    "creature","artifact creature","planeswalker","instant","sorcery",
    "enchantment","saga","battle","land","basic land","artifact","kindred"
]


def build_skeleton_params(cfg: Dict[str, Any], mode: str) -> SkeletonParams:
    sp_cfg = cfg['skeleton_params']
    # Force mode
    sp_cfg['types_mode'] = mode
    # After normalization, card_types_weights already injected
    return SkeletonParams(
        canonical_card_types=sp_cfg['canonical_card_types'],
        colors=sp_cfg['colors'],
        colors_weights=sp_cfg['colors_weights'],
        mana_values=sp_cfg['mana_values'],
        mana_curves=sp_cfg['mana_curves'],
        color_bleed_factor=sp_cfg['color_bleed_factor'],
        land_color_bleed_overlinear=sp_cfg['land_color_bleed_overlinear'],
        legend_mutation_factor=sp_cfg['legend_mutation_factor'],
        type_mutation_factor=sp_cfg['type_mutation_factor'],
        wildcard_mutation_factor=sp_cfg['wildcard_mutation_factor'],
        wildcard_supertype=sp_cfg['wildcard_supertype'],
        rarity_based_mutation=sp_cfg['rarity_based_mutation'],
        card_types=sp_cfg['canonical_card_types'],
        card_types_weights=sp_cfg['card_types_weights'],
        rarities_weights=sp_cfg['rarities_weights'],
        function_tags=sp_cfg['function_tags'],
        tags_maximum=sp_cfg['tags_maximum'],
        mutation_chance_per_theme=sp_cfg['mutation_chance_per_theme'],
        fixed_amount_themes=sp_cfg['fixed_amount_themes'],
        power_level=sp_cfg['power_level'],
        rarity_to_skew=sp_cfg['rarity_to_skew'],
        types_mode=mode,
    )


def _sample_types_once(sp: SkeletonParams) -> list[str]:
    """Replicate type selection + mutation loop similar to card_skeleton_generator."""
    colors = sp.colors
    color = random.choices(colors, weights=sp.colors_weights, k=1)[0]
    card_types = sp.card_types
    weights_row = sp.card_types_weights[color]
    initial = random.choices(card_types, weights=weights_row, k=1)[0]
    selected = [initial]
    basic_land_flag = initial.lower() == "basic land"
    primary_land_flag = initial.lower() == "land"
    t_chance = sp.type_mutation_factor
    # Replacement mutation
    if (not basic_land_flag and not primary_land_flag and check_mutation(t_chance)):
        selected = [random.choice(card_types)]
    # Additional mutations
    available = [t for t in card_types if t not in selected]
    while (not basic_land_flag and available and check_mutation(t_chance)):
        m = random.choice(available)
        selected.append(m)
        available.remove(m)
    return selected


def simulate(cfg: Dict[str, Any], mode: str, decks: int, deck_size: int, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    # Zero out sleepy time for speed
    cfg['square_config']['sleepy_time'] = 0

    sp = build_skeleton_params(cfg, mode)
    api = APIParams(api_key="DUMMY", auth_token="DUMMY", setParams=cfg.get('set_params', {}))

    deck_type_totals = []
    global_counter = Counter()
    for _ in range(decks):
        ctr = Counter()
        for _ in range(deck_size):
            types = _sample_types_once(sp)
            for t in types:
                ctr[t] += 1
        deck_type_totals.append(ctr)
        global_counter.update(ctr)

    # Compute averages & percentages
    avg_per_deck = {t: sum(c.get(t,0) for c in deck_type_totals)/decks for t in PRIMARY_TYPES}
    pct = {t: (global_counter.get(t,0)/(decks*deck_size))*100 for t in PRIMARY_TYPES}
    return {
        'mode': mode,
        'decks': decks,
        'deck_size': deck_size,
        'avg_per_deck': avg_per_deck,
        'percentages': pct,
        'global_counts': dict(global_counter)
    }


def print_report(res: Dict[str, Any]):
    print(f"Mode: {res['mode']}  Decks: {res['decks']}  Deck Size: {res['deck_size']}")
    print("Type Distribution (avg / %):")
    for t in PRIMARY_TYPES:
        avg = res['avg_per_deck'][t]
        pct = res['percentages'][t]
        if avg == 0 and pct == 0:
            continue
        print(f"  {t:15s} avg {avg:5.2f}  ({pct:5.2f}%)")
    total_pct = sum(res['percentages'].values())
    print(f"Total %: {total_pct:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='Optional user config path. If omitted with --empty uses defaults only.')
    ap.add_argument('--modes', nargs='*', default=['normal','play'])
    ap.add_argument('--decks', type=int, default=100)
    ap.add_argument('--deck-size', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--empty', action='store_true', help='Use only defaults (create ephemeral minimal config).')
    args = ap.parse_args()

    cfg_path = args.config
    if args.empty or not cfg_path:
        tmp_path = Path('configs/__tmp_sim.yml')
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write('# auto-generated minimal simulation config\n')
            f.write('skeleton_params:\n')
            f.write('  types_mode: normal\n')
        cfg_path = str(tmp_path)

    normalized = check_and_normalize_config(cfg_path, save=False)
    if normalized is None:
        print('Failed to normalize config.')
        return
    cfg = normalized

    for mode in args.modes:
        # Override mode each loop (simulate different baseline sets)
        cfg['skeleton_params']['types_mode'] = mode
        if args.empty:
            # rewrite ephemeral file for normalization to rebuild type weights for this mode
            tmp_path = Path(cfg_path)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write('# auto-generated minimal simulation config\n')
                f.write('skeleton_params:\n')
                f.write(f'  types_mode: {mode}\n')
            cfg = check_and_normalize_config(str(tmp_path), save=False) or cfg
        res = simulate(cfg, mode, args.decks, args.deck_size, args.seed)
        print_report(res)
        print('-'*70)

if __name__ == '__main__':
    main()
