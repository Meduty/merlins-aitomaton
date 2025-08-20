"""
================================================================================
 MTG Card Generator
--------------------------------------------------------------------------------
 Generates Magic: The Gathering cards by building a prompt skeleton
 and calling the MTG Card Generator API, then polling for and logging results.
--------------------------------------------------------------------------------
 Author  : Merlin Duty-Knez
 Date    : July 28, 2025
================================================================================
"""

########## Improvements: Read only field implementing

import math

import requests
import time
import json
import logging
import threading
import random
from queue import Queue, Empty

import numpy as np

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from copy import copy, deepcopy

from dotenv import load_dotenv

try:
    from . import merlinAI_lib
    from . import config_manager
    from .metrics import GenerationMetrics
except ImportError:
    # When running directly (not as a module)
    import merlinAI_lib
    import config_manager
    from metrics import GenerationMetrics

from typing import Any, Dict, Optional

# Load environment variables from .env file
load_dotenv()

import os

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class APIParams:
    """
    Class to hold API parameters for card generation.
    """

    def __init__(
        self,
        api_key: str,
        auth_token: str,
        userPrompt: Optional[Dict[str, Any]] = None,  # should be dict
        setParams: Optional[Dict[str, Any]] = None,  # should be dict
        generate_image_prompt: bool = False,
        creative: bool = False,
        include_explanation: bool = False,
        image_model: str = "dall-e-3",  # dall-e-3, dall-e-2, none
        model: str = "gpt-41",
    ):
        self.api_key = api_key

        # Keep raw token (don’t reconstruct by parsing headers later)
        self.auth_token = auth_token
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        self.generate_image_prompt = bool(generate_image_prompt)
        self.creative = bool(creative)
        self.include_explanation = bool(include_explanation)
        self.image_model = image_model
        self.model = model

        # Fresh dicts to avoid shared state across instances
        self.userPrompt: Dict[str, Any] = dict(userPrompt) if userPrompt else {}
        self.setParams: Dict[str, Any] = dict(setParams) if setParams else {}

    # -------- copying --------

    def __copy__(self) -> "APIParams":
        """
        Shallow copy: nested dicts are shallow-copied.
        """
        return APIParams(
            api_key=self.api_key,
            auth_token=self.auth_token,
            userPrompt=copy(self.userPrompt),
            setParams=copy(self.setParams),
            generate_image_prompt=self.generate_image_prompt,
            creative=self.creative,
            include_explanation=self.include_explanation,
            image_model=self.image_model,
            model=self.model,
        )

    def __deepcopy__(self, memo) -> "APIParams":
        """
        Deep copy: nested structures are fully copied.
        """
        return APIParams(
            api_key=deepcopy(self.api_key, memo),
            auth_token=deepcopy(self.auth_token, memo),
            userPrompt=deepcopy(self.userPrompt, memo),
            setParams=deepcopy(self.setParams, memo),
            generate_image_prompt=self.generate_image_prompt,
            creative=self.creative,
            include_explanation=self.include_explanation,
            image_model=self.image_model,
            model=self.model,
        )

    # -------- output / serialization --------

    def params_out(self) -> Dict[str, Any]:
        """
        Produce a safe dict for logging/transport.
        Redacts the API key; omits empty payloads.
        """

        def redact(key: str) -> str:
            if not key:
                return ""
            if len(key) <= 6:
                return "***"
            return f"{key[:3]}...{key[-3:]}"

        out: Dict[str, Any] = {
            "openAIApiKey": redact(self.api_key),
            "generateImagePrompt": self.generate_image_prompt,
            "extraCreative": self.creative,
            "includeExplanation": self.include_explanation,
            "imageModel": self.image_model,
            "model": self.model,
        }
        if self.userPrompt:
            out["userPrompt"] = self.userPrompt
        return out

    # -------- mutations --------

    def update_auth_token(self, new_auth_token: str, sleepy_time: float = 0) -> None:
        """
        Update the authorization token in the headers.
        Note: In multi-threaded environments, this should be called within a lock
        to prevent race conditions when multiple threads attempt to update simultaneously.
        """
        self.auth_token = new_auth_token
        self.headers["Authorization"] = f"Bearer {new_auth_token}"
        logging.info("Authorization token updated successfully.")
        time.sleep(sleepy_time)

    # -------- convenience --------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "APIParams":
        """
        Construct from a dict (e.g., YAML). Extra keys are ignored.
        """
        allowed = {
            "api_key",
            "auth_token",
            "userPrompt",
            "setParams",
            "generate_image_prompt",
            "creative",
            "include_explanation",
            "image_model",
            "model",
        }
        kwargs = {k: config[k] for k in config.keys() & allowed}
        return cls(**kwargs)


class SkeletonParams:
    """
    Class to hold parameters for card skeleton generation.
    """

    def __init__(
        self,
        canonical_card_types: Optional[list[str]] = None,
        colors: Optional[list[str]] = None,
        colors_weights: Optional[dict[str, float] | list[float]] = None,
        mana_values: Optional[list[str]] = None,
        mana_curves: Optional[dict[str, list[float]]] = None,
        color_bleed_factor: int = 20,  # in %
        land_color_bleed_overlinear: int = 2,  # overlinear factor for land color bleed
        legend_mutation_factor: int = 1,  # in % used for flat legendaries
        type_mutation_factor: int = 10,  # in %
        wildcard_mutation_factor: int = 5,  # in %
        wildcard_supertype: bool = False,
        rarity_based_mutation: Optional[dict[str, list[int]]] = None,
        card_types: Optional[list[str]] = None,
        card_types_weights: Optional[dict[str, dict[str, float] | list[float]]] = None,
        rarities: Optional[list[str]] = None,
        rarities_weights: Optional[dict[str, float] | list[float]] = None,
        function_tags: Optional[dict[str, int]] = None,  # each in percent
        tags_maximum: Optional[int] = None,  # maximum number of function tags to apply
        mutation_chance_per_theme: int = 20,  # in %
        fixed_amount_themes: int = 1,  # if not zero, select fixed amount instead of random mutation
        power_level: float = 5,  # Power level of the card, 1–10
        rarity_to_skew: Optional[dict[str, int]] = None,  # rarity skew mapping
    ):
        # Set up canonical card types and default weights from config - NO FALLBACKS
        if canonical_card_types is None:
            raise ValueError("canonical_card_types must be provided in configuration")
        if colors is None:
            raise ValueError("colors must be provided in configuration")
        
        # Extract default type weights from card_types_weights._default
        if card_types_weights is None or "_default" not in card_types_weights:
            raise ValueError("card_types_weights._default must be provided in configuration")
        
        default_type_weights = card_types_weights["_default"]
        if not isinstance(default_type_weights, dict):
            raise ValueError("card_types_weights._default must be a dictionary")
        
        self.canonical_card_types = canonical_card_types
        self.default_type_weights = default_type_weights
        self.colors = colors

        cw = colors_weights
        default_cw = {c: 100.0 / len(self.colors) for c in self.colors}
        if isinstance(cw, list):
            cw = {c: float(cw[i]) for i, c in enumerate(self.colors[: len(cw)])}
        cw = cw or {}
        merged_cw = {
            **default_cw,
            **{k: float(v) for k, v in cw.items() if k in self.colors},
        }
        total = sum(merged_cw.values()) or 1.0
        self.colors_weights_dict = {
            c: (v * 100.0 / total) for c, v in merged_cw.items()
        }
        self.colors_weights = [self.colors_weights_dict[c] for c in self.colors]

        if mana_values is None:
            raise ValueError("mana_values must be provided in configuration")
        self.mana_values = mana_values
        
        if mana_curves is None:
            raise ValueError("mana_curves must be provided in configuration")
        self.mana_curves = mana_curves

        self.color_bleed_factor = color_bleed_factor
        self.land_color_bleed_overlinear = land_color_bleed_overlinear
        self.legend_mutation_factor = legend_mutation_factor
        self.type_mutation_factor = type_mutation_factor
        self.wildcard_mutation_factor = wildcard_mutation_factor
        self.wildcard_supertype = wildcard_supertype
        
        if rarity_based_mutation is None:
            raise ValueError("rarity_based_mutation must be provided in configuration")
        self.rarity_based_mutation = rarity_based_mutation

        if card_types is None:
            card_types = self.canonical_card_types
        self.card_types = card_types
        
        self.card_types_weights = self._build_type_weights(
            card_types=self.card_types,
            weights_by_color=card_types_weights or {},
            colors=self.colors,
            code_defaults=self.default_type_weights,
        )

        if rarities is None:
            raise ValueError("rarities must be provided in configuration")
        self.rarities = rarities
        rw = rarities_weights
        default_rw = {r: 100.0 / len(self.rarities) for r in self.rarities}
        if isinstance(rw, list):
            rw = {r: float(rw[i]) for i, r in enumerate(self.rarities[: len(rw)])}
        rw = rw or {}
        merged_rw = {
            **default_rw,
            **{k: float(v) for k, v in rw.items() if k in self.rarities},
        }
        total_rw = sum(merged_rw.values()) or 1.0
        self.rarities_weights_dict = {
            r: (v * 100.0 / total_rw) for r, v in merged_rw.items()
        }
        self.rarities_weights = [self.rarities_weights_dict[r] for r in self.rarities]

        if function_tags is None:
            raise ValueError("function_tags must be provided in configuration")
        self.function_tags = function_tags

        self.tags_maximum = tags_maximum
        self.mutation_chance_per_theme = mutation_chance_per_theme
        self.fixed_amount_themes = fixed_amount_themes
        self.power_level = power_level
        
        # Rarity to skew mapping with defaults
        if rarity_to_skew is None:
            raise ValueError("rarity_to_skew must be provided in configuration")
        self.rarity_to_skew = rarity_to_skew

    @staticmethod
    def _normalize_row_to_sum(row, total=100.0):
        s = sum(row)
        if s > 0:
            f = total / s
            return [x * f for x in row]
        return row

    @classmethod
    def _build_type_weights(
        cls,
        *,
        card_types: list[str],
        weights_by_color: dict,
        colors: list[str],
        code_defaults: dict[str, float],
        normalize: bool = True,
    ) -> dict[str, list[float]]:
        """
        Produce dict[color -> list[float]] aligned to `card_types`.

        Precedence per color/type:
          1) per-color override in YAML (dict or legacy list)
          2) YAML `_default` row
          3) code_defaults (if YAML `_default` missing or missing keys)

        Always returns every color in `colors`.
        Accepts legacy list rows (zipped to card_types).
        """

        # --- Warn if YAML contains unknown colors ---
        unknown_colors = set(weights_by_color.keys()) - {"_default"} - set(colors)
        if unknown_colors:
            logging.warning(
                f"Unknown colors in card_types_weights ignored: {sorted(unknown_colors)}"
            )

        # 1) Build the baseline `_default` row map
        yaml_default = weights_by_color.get("_default", {})
        if isinstance(yaml_default, list):
            # allow legacy list for _default too
            default_map = {
                t: float(yaml_default[i])
                for i, t in enumerate(card_types[: len(yaml_default)])
            }
        elif isinstance(yaml_default, dict):
            default_map = {
                t: float(yaml_default.get(t, code_defaults.get(t, 0.0)))
                for t in card_types
            }
        else:
            default_map = {t: float(code_defaults.get(t, 0.0)) for t in card_types}

        def build_row_from_source(src):
            """src can be dict[type->w] or legacy list; merged over default_map."""
            if src is None:
                wmap = dict(default_map)
            elif isinstance(src, list):
                wmap = dict(default_map)
                for i, t in enumerate(card_types[: len(src)]):
                    wmap[t] = float(src[i])
            elif isinstance(src, dict):
                wmap = dict(default_map)
                unknown_types = set(src.keys()) - set(card_types)
                if unknown_types:
                    logging.warning(
                        f"Unknown types ignored in row: {sorted(unknown_types)}"
                    )
                for t, w in src.items():
                    if t in wmap:
                        wmap[str(t)] = float(w)
            else:
                raise TypeError("card_types_weights entries must be dict or list")
            row = [wmap.get(t, default_map.get(t, 0.0)) for t in card_types]
            return cls._normalize_row_to_sum(row) if normalize else row

        # 2) Build rows for all canonical colors
        result: dict[str, list[float]] = {}
        for color in colors:
            src = weights_by_color.get(color)  # may be dict, list, or None
            result[color] = build_row_from_source(src)

        return result


def login_mtgcg() -> str:
    """
    Login to the MTG Card Generator API using environment variables.
    Returns:
        str: Access token for the MTG Card Generator API.
    """
    username = os.getenv("MTGCG_USERNAME")
    password = os.getenv("MTGCG_PASSWORD")

    if not username or not password:
        raise ValueError(
            "MTGCG_USERNAME and MTGCG_PASSWORD must be set in environment variables."
        )

    url = "https://mtgcardgenerator.azurewebsites.net/api/Authenticate"
    payload = {"username": username, "password": password}

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        access_token = data.get("accessToken")
        if not access_token:
            raise Exception("No accessToken returned from authentication API.")
        return access_token
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        raise


def bounded_value_with_rarity(
    mean: float,
    low: float,
    high: float,
    *,
    sd: float = 0.7,
    rarity: str | None = None,
    rarity_skew: float = 0.35,  # 0..1 → how much rarity influences
    rng: np.random.Generator | None = None,
    rarity_to_skew: dict[str, int] | None = None,
) -> float:
    """
    Convex-combine:
        value = (1 - rarity_skew) * TruncNormal(mean, sd) + rarity_skew * BetaSkewed(rarity)

    - Keeps results bounded in [low, high]
    - Anchored around user's `mean`
    - Nudged up/down by rarity via Beta skew
    """
    assert 0.0 <= rarity_skew <= 1.0, "rarity_skew must be in [0,1]"
    if rng is None:
        rng = np.random.default_rng()

    tn = merlinAI_lib.truncated_normal_random(mean, sd, low, high)  # rng=rng ?
    if rarity is None:
        return tn

    # Use provided rarity_to_skew mapping or default
    if rarity_to_skew is None:
        rarity_to_skew = {
            "common": -2,
            "uncommon": -1,
            "rare": 1,
            "mythic": 2,
        }
    
    skew = rarity_to_skew.get(str(rarity).lower(), 0)
    be = merlinAI_lib.beta_skewed_random(low, high, skew=skew, rng=rng)
    return (1.0 - rarity_skew) * tn + rarity_skew * be  # convex combo stays in-bounds


def chance_advantage(input_bleed, steigung=1) -> float:
    """
    Adjusts the color bleed factor for land cards.
    Land cards typically have a lower color bleed factor.
    """

    assert (
        input_bleed >= 0 and input_bleed <= 100
    ), "Color bleed factor must be between 0 and 100."

    res = math.pow(100, (steigung - 1) / steigung) * math.pow(input_bleed, 1 / steigung)

    # 100^(((c-1)/(c))) x^(((1)/(c)))

    return res


def card_skeleton_generator(
    index, api_params: APIParams, skeleton_params: SkeletonParams, config: Dict[str, Any]
) -> APIParams:
    """
    Generates a card skeleton with fixed values and random attributes.
    Config is passed as argument instead of using global variables.
    """
    # Extract config values instead of using globals
    sleepy_time = config["square_config"]["sleepy_time"]
    stdDePL = config["square_config"]["standard_deviation_powerLevel"]
    powSkew = config["square_config"]["power_level_rarity_skew"]

    out_params = deepcopy(api_params)

    card_skeleton = deepcopy(
        api_params.setParams
    )  # Copy set parameters for the skeleton
    logging.info(f"Generating card skeleton for index {index}")
    time.sleep(sleepy_time)

    ############## Set dynamic values for each card

    # Base color identity
    colors = skeleton_params.colors
    colors_weights = skeleton_params.colors_weights
    selected_colors = random.choices(colors, weights=colors_weights, k=1)
    logging.debug(f"[Card #{index+1}] Selected base color identity: {selected_colors}")

    # Types
    t = skeleton_params.card_types.copy()
    card_types_weights = skeleton_params.card_types_weights[selected_colors[0]]
    selected_types = random.choices(t, weights=card_types_weights, k=1)
    logging.debug(f"[Card #{index+1}] Selected type: {selected_types[0]}")

    t_chance = skeleton_params.type_mutation_factor

    basic_land_flag = False
    primary_land_flag = False
    spell_flag = False

    # Raise a basic land flag, if the selected type is basic land
    if selected_types[0].lower() == "basic land":
        time.sleep(sleepy_time)
        logging.info(f"[Card #{index+1}] Selected type is basic land, setting flag.")
        basic_land_flag = True
    elif selected_types[0].lower() == "land":
        time.sleep(sleepy_time)
        logging.info(
            f"[Card #{index+1}] Selected type is land, setting primary land flag."
        )
        primary_land_flag = True
    elif (
        selected_types[0].lower() == "sorcery" or selected_types[0].lower() == "instant"
    ):
        logging.info(
            f"[Card #{index+1}] Selected type is {selected_types[0]}, setting spell flag."
        )
        spell_flag = True

    # Change types based on mutation chance
    if (
        not basic_land_flag
        and not primary_land_flag
        and merlinAI_lib.check_mutation(t_chance)
    ):
        logging.info(f"[Card #{index+1}] Type mutation occurred.")
        time.sleep(sleepy_time)
        selected_types = [random.choice(t)]
        logging.debug(
            f"[Card #{index+1}] Newly selected type after mutation: {selected_types[0]}"
        )

    # Remove already selected types from t to avoid duplicates
    available_types = [typ for typ in t if typ not in selected_types]

    # Add additional types based on mutation chance
    while (
        not basic_land_flag
        and available_types
        and merlinAI_lib.check_mutation(t_chance)
    ):
        logging.debug(f"[Card #{index+1}] Type mutation occurred.")
        mutation = random.choice(available_types)
        selected_types.append(mutation)
        available_types.remove(mutation)
        logging.debug(
            f"[Card #{index+1}] Added mutation type: {mutation}, new types: {selected_types}"
        )

    new_type = ", ".join(selected_types)

    logging.info(f"[Card #{index+1}] Final type: {new_type}")
    time.sleep(sleepy_time)

    card_skeleton["type"] = new_type

    # Mana value for non-land cards
    selected_mana_value = None
    if not basic_land_flag and not primary_land_flag:
        mana_values = skeleton_params.mana_values
        curve = skeleton_params.mana_curves.get(
            selected_colors[0],
            skeleton_params.mana_curves["default"]
        )
        selected_mana_value = random.choices(mana_values, weights=curve, k=1)[0]
        logging.info(f"[Card #{index+1}] Selected mana value: {selected_mana_value}")
        time.sleep(sleepy_time)
        card_skeleton["manaValue"] = selected_mana_value

    # Bleeding colors
    c = [
        col
        for col in skeleton_params.colors
        if col not in selected_colors and col != "colorless"
    ]
    color_bleed_factor = skeleton_params.color_bleed_factor

    if primary_land_flag:
        land_color_bleed_overlinear = skeleton_params.land_color_bleed_overlinear
        logging.info(
            f"[Card #{index+1}] Land color bleed factor: {land_color_bleed_overlinear}, "
            f"original bleed chance: {color_bleed_factor}"
        )
        time.sleep(sleepy_time)
        color_bleed_factor = chance_advantage(
            color_bleed_factor, steigung=land_color_bleed_overlinear
        )
        logging.info(f"[Card #{index+1}] New bleed chance: {color_bleed_factor}")
        time.sleep(sleepy_time)

    while (
        not basic_land_flag
        and selected_colors[0] != "colorless"
        and merlinAI_lib.check_mutation(color_bleed_factor)
    ):
        logging.debug(f"[Card #{index+1}] Color bleed mutation occurred.")
        if not c:
            logging.debug(f"[Card #{index+1}] No colors left for bleed mutation.")
            break

        # Build weights for available colors
        weights = [skeleton_params.colors_weights_dict[col] for col in c]

        logging.debug(f"[Card #{index+1}] Adding color bleed to {selected_colors}.")
        logging.debug(
            f"[Card #{index+1}] Available colors for bleed: {c} with weights {weights}"
        )

        bleed_color = random.choices(c, weights=weights, k=1)[0]
        selected_colors.append(bleed_color)
        c.remove(bleed_color)

        logging.debug(
            f"[Card #{index+1}] Added bleed color: {bleed_color}, new colors: {selected_colors}"
        )

    logging.info(f"[Card #{index+1}] Final color identity: {selected_colors}")
    time.sleep(sleepy_time)
    card_skeleton["colorIdentity"] = ", ".join(selected_colors)

    # Rarity
    rarities = skeleton_params.rarities
    rarity_weights = skeleton_params.rarities_weights
    selected_rarity = random.choices(rarities, weights=rarity_weights, k=1)[0]
    logging.info(f"[Card #{index+1}] Selected rarity: {selected_rarity}")
    time.sleep(sleepy_time)
    card_skeleton["rarity"] = selected_rarity

    # Legendary
    supertypes = []
    if not basic_land_flag and not spell_flag and merlinAI_lib.check_mutation(
        skeleton_params.legend_mutation_factor
    ):
        supertypes.append("Legendary")
        logging.debug(f"[Card #{index+1}] Added legendary supertype.")
    elif not basic_land_flag and not spell_flag and skeleton_params.rarity_based_mutation:
        a, b = skeleton_params.rarity_based_mutation[selected_rarity]
        den = a + b
        legend_chance = (a / den * 100.0) if den > 0 else 0.0
        logging.debug(
            f"[Card #{index+1}] Legendary mutation chance: {legend_chance:.2f}"
        )
        if merlinAI_lib.check_mutation(legend_chance):
            supertypes.append("Legendary")
            logging.debug(
                f"[Card #{index+1}] Added legendary supertype based on rarity mutation."
            )

    # WILDCARD!
    if skeleton_params.wildcard_supertype and merlinAI_lib.check_mutation(
        skeleton_params.wildcard_mutation_factor
    ):
        supertypes.append("Wildcard")
        logging.debug(f"[Card #{index+1}] WILDCARD! Added wildcard supertype.")

    if supertypes:
        card_skeleton["supertypes"] = ", ".join(supertypes)
        logging.info(
            f"[Card #{index+1}] Added supertypes: {card_skeleton['supertypes']}"
        )
        time.sleep(sleepy_time)

    # Extra creative
    a, b = skeleton_params.rarity_based_mutation[selected_rarity]
    den = a + b
    extra_creative_chance = (a / den * 100.0) if den > 0 else 0.0
    if merlinAI_lib.check_mutation(extra_creative_chance):
        out_params.creative = True
        logging.info(f"[Card #{index+1}] Extra creative!")
        time.sleep(sleepy_time)

    # function tags
    function_tags = skeleton_params.function_tags
    selected_tags = []
    logging.debug(
        f"[Card #{index+1}] Checking for function tags. Available tags: {function_tags}"
    )
    for tag, chance in function_tags.items():
        logging.debug(
            f"[Card #{index+1}] Checking special tag: {tag} with chance: {chance}"
        )
        if merlinAI_lib.check_mutation(chance):
            selected_tags.append(tag)
            logging.debug(f"[Card #{index+1}] Added special tag: {tag}")

    logging.info(f"[Card #{index+1}] Selected function tags: {selected_tags}")
    time.sleep(sleepy_time)

    otag_max = skeleton_params.tags_maximum

    if basic_land_flag:
        logging.info(f"[Card #{index+1}] Basic land card, no function tags allowed.")
        time.sleep(sleepy_time)
        otag_max = None
    elif primary_land_flag:
        selected_tags.append("Simple or straightforward, no complexities")

    if otag_max is not None:
        assert (
            isinstance(otag_max, int) and otag_max >= 0
        ), "tags_maximum must be a non-negative int."
        logging.debug(f"[Card #{index+1}] Maximum function tags allowed: {otag_max}")
        if len(selected_tags) > otag_max:
            logging.info(
                f"[Card #{index+1}] Selected tags exceed maximum, trimming to {otag_max}"
            )
            time.sleep(sleepy_time)
            selected_tags = random.sample(selected_tags, otag_max)

    if selected_tags:
        card_skeleton["function_tags"] = ", ".join(selected_tags)
        logging.info(
            f"[Card #{index+1}] Added function tags: {card_skeleton['function_tags']}"
        )
        time.sleep(sleepy_time)

    if basic_land_flag:
        card_skeleton["function_tags"] = "Basic Land, SHOULD NOT HAVE ANY ABILITIES"

    # Card power level
    if skeleton_params.power_level is not None:
        assert isinstance(
            skeleton_params.power_level, (int, float)
        ), "Power level must be a number."
        assert (
            1.0 <= float(skeleton_params.power_level) <= 10.0
        ), "Power level must be between 1 and 10."

        powerLevel = skeleton_params.power_level
        powerLevel = bounded_value_with_rarity(
            mean=powerLevel,
            low=1,
            high=10,
            sd=stdDePL,
            rarity=selected_rarity,
            rarity_skew=powSkew,  # Adjust this value as needed
            rarity_to_skew=skeleton_params.rarity_to_skew,
        )
        powerLevel = round(powerLevel, 2)  # Round to 2 decimal places
        card_skeleton["powerLevel"] = f"{powerLevel} out of 10"
        logging.info(
            f"[Card #{index+1}] Set power level to {skeleton_params.power_level}, adjusted to {powerLevel}."
        )
        time.sleep(sleepy_time)

        # PATCH: replace the “Themes” header block
        logging.info(f"[Card #{index+1}] Checking for themes.")
        themes = list(card_skeleton.get("themes", []))
        logging.debug(f"[Card #{index+1}] Available themes: {themes}")
        time.sleep(sleepy_time)

        selected_themes = []
        th = themes.copy()
        amount = skeleton_params.fixed_amount_themes

        if isinstance(amount, int) and amount > 0:
            logging.debug(f"[Card #{index+1}] Fixed amount of themes: {amount}")
            if amount > len(th):
                logging.warning(
                    f"[Card #{index+1}] Requested amount exceeds available themes, adjusting to {len(th)}"
                )
                amount = len(th)
            selected_themes = random.sample(th, amount)
            logging.info(f"[Card #{index+1}] Selected themes: {selected_themes}")
            time.sleep(sleepy_time)
        else:
            for theme in th:
                logging.debug(f"[Card #{index+1}] Checking theme: {theme}")
                logging.debug(
                    f"[Card #{index+1}] Theme weight: {skeleton_params.mutation_chance_per_theme}"
                )
                if merlinAI_lib.check_mutation(
                    skeleton_params.mutation_chance_per_theme
                ):
                    selected_themes.append(theme)
                    logging.debug(f"[Card #{index+1}] Added theme: {theme}")

        card_skeleton["themes"] = selected_themes
        logging.info(f"[Card #{index+1}] Selected themes: {selected_themes}")
        time.sleep(sleepy_time)

    ########### Logging the generated card skeleton
    logging.debug(
        f"[Card #{index+1}] Card skeleton generated:\n"
        f"{json.dumps(card_skeleton, indent=2)}"
    )

    out_params.userPrompt = card_skeleton  # Update the user prompt in API params

    # Add a sleep to slow down skeleton generation if needed
    time.sleep(sleepy_time)
    return out_params


def generate_card(index, api_params: APIParams, metrics: GenerationMetrics, config: Dict[str, Any]) -> dict:
    """API parameters with metrics tracking and config passed as arguments."""
    sleepy_time = config["square_config"]["sleepy_time"]
    timeout = config["http_config"]["timeout"]
    polling_interval = config["http_config"]["polling_interval"]

    local_api_params = copy(api_params)

    card_start = time.time()
    logging.info(f"[#{index+1}] Send card generation request...")
    time.sleep(sleepy_time)

    params = {
        "openAIApiKey": local_api_params.api_key,
        "generateImagePrompt": local_api_params.generate_image_prompt,
        "extraCreative": local_api_params.creative,
        "includeExplanation": local_api_params.include_explanation,
        "imageModel": local_api_params.image_model,
        "model": local_api_params.model,
        "userPrompt": json.dumps(
            local_api_params.userPrompt, separators=(',', ':')
        ),  # Use compact JSON to minimize URL length
    }

    if params["imageModel"] == "none":
        # Delete openAIApiKey if imageModel is none
        del params["openAIApiKey"]

    auth = local_api_params.headers

    logging.debug(f"[#{index+1}] Request parameters: {json.dumps(params, indent=2)}")

    try:
        logging.info(f"[#{index+1}] Sending request...")
        time.sleep(sleepy_time)
        
        # Check URL length to warn about potential 414 errors
        url_with_params = requests.Request('GET', 
            "https://mtgcardgenerator.azurewebsites.net/api/GenerateMagicCard",
            params=params
        ).prepare().url
        
        if len(url_with_params) > 2000:  # Conservative limit
            logging.warning(f"[#{index+1}] Long URL detected ({len(url_with_params)} chars) - may cause HTTP 414 error")
        resp = requests.get(
            "https://mtgcardgenerator.azurewebsites.net/api/GenerateMagicCard",
            headers=auth,
            params=params,
            timeout=timeout
        )
        logging.debug(f"[#{index+1}] server response: {resp.status_code}")
        resp.raise_for_status()
        card_id = resp.json().get("id")
        if not card_id:
            logging.error(f"[#{index+1}] No card ID returned, possibly rejected.")
            raise Exception("No card ID returned from API response.")
    except Exception as e:
        logging.error(f"[#{index+1}] Card generation failed: {e}")
        raise Exception("Card generation failed") from e

    logging.info(f"[#{index+1}] Received card ID: {card_id}")
    time.sleep(sleepy_time)
    logging.info(f"[#{index+1}] Now Polling card generation status ...")
    time.sleep(sleepy_time)

    status_url = f"https://mtgcardgenerator.azurewebsites.net/api/GetMagicCardGenerationStatus?instanceId={card_id}"
    while True:
        try:
            status_resp = requests.get(status_url, headers=auth, timeout=timeout)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            logging.debug(
                f"[#{index+1}] Polling status: {status_data['runtimeStatus']}"
            )
            if status_data["runtimeStatus"] == "Completed":
                break
            elif (
                status_data["runtimeStatus"] != "Running"
                and status_data["runtimeStatus"] != "Pending"
            ):
                logging.error(
                    f"[#{index+1}] Unexpected status: {status_data['runtimeStatus']}"
                )
                raise Exception(f"Unexpected status: {status_data['runtimeStatus']}")
            elif time.time() - card_start > timeout:  # Timeout after {timeout} seconds
                logging.error(
                    f"[#{index+1}] Card generation timed out after {timeout} seconds."
                )
                raise TimeoutError(f"Card generation timed out after {timeout} seconds.")
            time.sleep(polling_interval)

        except Exception as e:
            logging.error(f"[#{index+1}] Polling failed: {e}")
            raise Exception("Polling failed") from e

    output_json_str = status_data.get("output", "")
    if not output_json_str:
        logging.error(f"[#{index+1}] Empty or missing output for card ID {card_id}")
        logging.error(f"[#{index+1}] output: {output_json_str}")
        raise ValueError("Empty or missing output from card generation.")

    try:
        output_data = json.loads(output_json_str)
    except json.JSONDecodeError as e:
        logging.error(f"[#{index+1}] JSON decode error for card ID {card_id}: {e}")
        logging.error(f"[#{index+1}] output: {output_json_str}")
        raise ValueError(f"JSON decode error for card ID {card_id}: {e}")

    try:
        color_identity = output_data["cards"][0].get("colorIdentity", "").lower()
        generated_rarity = output_data["cards"][0].get("rarity", "").lower()
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"[#{index+1}] Error accessing card data for card ID {card_id}: {e}")
        logging.error(f"[#{index+1}] Available keys in output_data: {list(output_data.keys()) if isinstance(output_data, dict) else 'Not a dict'}")
        if isinstance(output_data, dict) and "cards" in output_data:
            logging.error(f"[#{index+1}] Cards array length: {len(output_data['cards']) if isinstance(output_data['cards'], list) else 'Not a list'}")
            if isinstance(output_data["cards"], list) and len(output_data["cards"]) > 0:
                logging.error(f"[#{index+1}] First card keys: {list(output_data['cards'][0].keys()) if isinstance(output_data['cards'][0], dict) else 'Not a dict'}")
        logging.error(f"[#{index+1}] Full output_data structure: {json.dumps(output_data, indent=2)}")
        raise ValueError(f"Invalid card data structure for card ID {card_id}: {e}")

    # Update metrics using the passed metrics object instead of global variables
    metrics.update_color(color_identity)
    metrics.update_rarity(generated_rarity)

    card_runtime = time.time() - card_start
    metrics.add_runtime(card_runtime)
    metrics.increment_successful()
    
    logging.info(
        f"[Card #{index+1}] Color: {color_identity.title()}, Rarity: {generated_rarity.title()}, Time: {card_runtime:.2f}s"
    )
    time.sleep(sleepy_time)

    logging.info(f"[Card #{index+1}] Card generation successful.")
    time.sleep(sleepy_time)
    logging.debug(f"[Card #{index+1}] Card output data: {output_data}")

    return output_data


def get_card_graceful(i, api_params: APIParams, skeleton_params: SkeletonParams, 
                     metrics: GenerationMetrics, config: Dict[str, Any], 
                     retries=3, retry_delay=10, auth_lock=None) -> dict:
    """
    Wrapper to handle card generation with retries.
    """
    sleepy_time = config["square_config"]["sleepy_time"]

    logging.debug(
        f"[Card #{i+1}] User prompt: {api_params.userPrompt}, Creative: {api_params.creative}"
    )

    attempt = 1
    card = None
    while attempt <= retries:
        try:

            local_api_params = card_skeleton_generator(
                i, api_params=api_params, skeleton_params=skeleton_params, config=config
            )

            logging.info(
                f"[Card #{i+1}] API requests:\n"
                f"{json.dumps(local_api_params.params_out(), indent=2)}"
            )
            time.sleep(sleepy_time)

            card = generate_card(i, api_params=local_api_params, metrics=metrics, config=config)
            break
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                logging.error(f"[Card #{i+1}] Unauthorized (401). Attempting re-login.")
                try:
                    # Use lock to prevent multiple threads from updating auth token simultaneously
                    if auth_lock:
                        with auth_lock:
                            new_auth_token = login_mtgcg()
                            api_params.update_auth_token(new_auth_token, sleepy_time)
                            logging.info(f"[Card #{i+1}] Auth token updated under lock")
                    else:
                        new_auth_token = login_mtgcg()
                        api_params.update_auth_token(new_auth_token, sleepy_time)
                except Exception as login_err:
                    logging.error(f"[Card #{i+1}] Re-login failed: {login_err}")
                    break
            else:
                logging.error(f"[Card #{i+1}] Generation failed: {e}")
        except Exception as e:
            logging.error(f"[Card #{i+1}] Generation failed: {e}")
        attempt += 1
        if attempt <= retries:
            logging.info(
                f"[Card #{i+1}] Retrying in {retry_delay * attempt}s (attempt {attempt}/{retries})..."
            )
            time.sleep(sleepy_time)
            time.sleep(retry_delay * attempt)
    if not card:
        logging.error(
            f"[Card #{i+1}] Failed to generate card after {retries} attempts."
        )
        raise Exception(f"Failed to generate card #{i+1} after {retries} attempts.")

    logging.info(f"[Card #{i+1}] generation complete.")
    time.sleep(sleepy_time)
    logging.info(f"[Card #{i+1}] Card output: {json.dumps(card, indent=2)}")
    time.sleep(sleepy_time)
    return card


################# Main Execution #################


def card_worker(card_queue, pbar, api_params, skeleton_params, metrics, config, max_retries, retry_delay, auth_lock):
    """Worker function for threaded card generation."""
    while True:
        try:
            i = card_queue.get_nowait()
        except Empty:
            break

        try:
            card = get_card_graceful(
                i, api_params=api_params, skeleton_params=skeleton_params,
                metrics=metrics, config=config, retries=max_retries, retry_delay=retry_delay,
                auth_lock=auth_lock
            )
            metrics.add_card(card["cards"][0])

        except Exception as e:
            logging.error(f"[Card #{i+1}] failed: {e}")
        finally:
            card_queue.task_done()
            pbar.update(1)  # <- keeps the bar in sync
            sleepy_time = config["square_config"]["sleepy_time"]
            logging.info(f"[Card #{i+1}] task done.")
            time.sleep(sleepy_time)


if __name__ == "__main__":
    # Parse command line arguments and load configuration
    args = config_manager.parse_args()
    config = config_manager.load_config(args.config)
    config = config_manager.apply_cli_overrides(config, args)
    
    # Extract configuration values
    total_cards = config["square_config"]["total_cards"]
    concurrency = config["square_config"]["concurrency"]
    max_retries = config["http_config"]["retries"]
    retry_delay = config["http_config"]["retry_delay"]
    sleepy_time = config["square_config"]["sleepy_time"]
    set_params = config["set_params"]
    
    # Load API credentials from environment variables
    API_KEY = os.getenv("API_KEY")
    AUTH_TOKEN = os.getenv("AUTH_TOKEN")
    
    # Progress bar format
    BAR_FMT = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[Elapsed: {elapsed} | Remaining: {remaining} | Avg: {rate_fmt}]"
    )
    
    # Initialize metrics tracking
    metrics = GenerationMetrics()
    
    # Create a lock for thread-safe auth token updates
    auth_lock = threading.Lock()

    card_skeleton_params = SkeletonParams(**config["skeleton_params"])
    api_params = APIParams(
        api_key=API_KEY,
        auth_token=AUTH_TOKEN,
        setParams=set_params,
        **config["api_params"],
    )

    if AUTH_TOKEN is None or AUTH_TOKEN == "None" or AUTH_TOKEN == "":
        logging.info("No auth token found, attempting to login...")
        time.sleep(sleepy_time)
        try:
            with auth_lock:
                new_auth_token = login_mtgcg()
                api_params.update_auth_token(new_auth_token, sleepy_time)
        except Exception as e:
            logging.error(f"Login failed: {e}")
            raise

    logging.info(f"[init] Set Params: {json.dumps(api_params.setParams, indent=2)}")
    time.sleep(sleepy_time)
    logging.info("=== Starting MTG Card Generation ===")
    time.sleep(sleepy_time)

    threads = []
    card_queue = Queue()
    for i in range(total_cards):
        card_queue.put(i)

    # Progress bar + logging that plays nicely with it
    with logging_redirect_tqdm():
        with tqdm(
            total=total_cards,
            desc="Generating card information",
            unit="card",
            bar_format=BAR_FMT,
        ) as pbar:
            for _ in range(min(concurrency, total_cards)):
                t = threading.Thread(
                    target=card_worker,
                    args=(
                        card_queue,
                        pbar,
                        api_params,
                        card_skeleton_params,
                        metrics,
                        config,
                        max_retries,
                        retry_delay,
                        auth_lock,
                    ),
                    daemon=False,
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

    # Get metrics summary
    summary = metrics.get_summary()

    # Final stats
    logging.info("=== Generation Complete ===")
    time.sleep(sleepy_time)
    logging.info(f"Rarity Distribution: {summary['rarities']}")
    time.sleep(sleepy_time)
    logging.info(f"Color Distribution: {summary['colors']}")
    time.sleep(sleepy_time)
    logging.info(f"Total Cards Generated: {summary['successful']}/{total_cards}")
    time.sleep(sleepy_time)
    logging.info(f"Total Runtime: {summary['total_runtime']:.2f}s")
    time.sleep(sleepy_time)
    logging.info(f"Avg Time per Card: {summary['average_time_per_card']:.2f}s")
    time.sleep(sleepy_time)
    logging.info("=== End of Generation ===")

    outdir = config["square_config"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, "generated_cards.json")

    with open(outname, "w") as f:
        json.dump(metrics.all_cards, f, indent=2)
        logging.info(f"Generated cards saved to {outname}")
        time.sleep(sleepy_time)
    logging.info("All threads completed successfully.")
    time.sleep(sleepy_time)
