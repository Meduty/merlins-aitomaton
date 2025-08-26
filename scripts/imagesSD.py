from datetime import datetime
import urllib.request
import base64
import json
import time
import os
import sys
from enum import Enum

import logging
import threading
from tqdm import tqdm

import random

import yaml

from typing import Any, Dict, List, Optional, Callable

from dotenv import load_dotenv

import numpy as np

# Handle imports for both direct execution and module import
try:
    from . import merlinAI_lib
except ImportError:
    # When running directly (not as a module)
    import merlinAI_lib

# Load environment variables from .env file
load_dotenv()


from openai import OpenAI

# Logging setup - respect orchestrator's verbose setting
def setup_logging():
    """Setup logging based on environment variable from orchestrator."""
    verbose = os.environ.get("MERLIN_VERBOSE", "1") == "1"
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True
        )
    else:
        # Suppress all logs except errors in quiet mode
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            force=True
        )

# Don't call setup_logging() at import time - let it be called when needed

# NOTE: Configuration now passed as parameters instead of global loading
# Global constants only
API_KEY = os.getenv("API_KEY")

# These are set by generate_images_from_dict based on config parameters
# NO DEFAULT VALUES - all must be provided in configuration for strict validation
forge_url_base = None
forge_out = None
random_lora_weights = None
apply_lora_chance = None
loraStDe = None
use_special_tags = None
vary_special_tags_weights = None
model_swap_chance = None
max_retries = None
retry_delay = None
max_tag_weight = None
sleepy_time = None
special_tags = None

# This will be set by generate_images_from_dict based on config
out_dir = None

BAR_FMT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[Elapsed: {elapsed} | Remaining: {remaining} | Avg: {rate_fmt}]"
)

class Model(Enum):
    AIDMA_IMAGE_UPRADER_FLUX_V0_3 = "aidmaImageUprader-FLUX-v0.3"
    ANYLORA_CHECKPOINT_BAKEDVAE_BLESSEDFP16 = "anyloraCheckpoint_bakedvaeBlessedFp16"
    EPICREALISM_XL_VXVII_CRYSTALCLEAR = "epicrealismXL_vxviiCrystalclear"
    FLUX1DEV_HYPERNF4_FLUX1DEVBNB_FLUX1DEVHYPERNF4 = (
        "flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4"
    )
    PONY_DIFFUSION_V6XL_V6START_WITH_THIS_ONE = "ponyDiffusionV6XL_v6StartWithThisOne"
    PONY_REALISM_V23ULTRA = "ponyRealism_V23ULTRA"
    PREFECT_ILLUSTRIOUS_XL_V20P = "prefectIllustriousXL_v20p"
    PREFECT_PONY_XL_V50 = "prefectPonyXL_v50"
    REAL_DREAM_SDXL_PONY15 = "realDream_sdxlPony15"
    REALISM_BY_STABLE_YOGI_V40FP16 = "realismByStableYogi_v40FP16"
    SDXL_UNSTABLE_DIFFUSERS_V9_DIVINITYMACHINEVAE = (
        "sdxlUnstableDiffusers_v9DIVINITYMACHINEVAE" 
    )
    WAI_NSFW_ILLUSTRIOUS_V110 = "waiNSFWIllustrious_v110"


# pbar helper
def _sd_progress_percent(timeout: float = 10.0) -> int:
    """Return Stable Diffusion percent [0..100], clamped, or 0 on error."""
    try:
        req = urllib.request.Request(f"{forge_url_base}/sdapi/v1/progress")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prog = float(data.get("progress", 0.0))
        return max(0, min(100, int(round(prog * 100))))
    except Exception:
        return 0

def resolve_model(d: Dict[str, Any]) -> Model:
    """Resolve model from enum NAME or enum VALUE. Strict."""
    if "model" in d:
        try:
            return Model[d["model"]]  # enum NAME
        except KeyError as e:
            raise ValueError(f"Unknown model enum name: {d['model']}") from e
    if "model_value" in d:
        for m in Model:
            if m.value == d["model_value"]:
                return m
        raise ValueError(f"Unknown model enum value: {d['model_value']}")
    raise ValueError("Missing 'model' or 'model_value' in option_params")


def load_image_options_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert config['image_options'] into validated options with Model enums."""
    options = config["SD_config"]["image_options"]  # strict: KeyError if missing
    out: List[Dict[str, Any]] = []

    if not isinstance(options, list) or not options:
        raise ValueError("'image_options' must be a non-empty list")

    for opt in options:
        name = opt["name"]
        weight = float(opt["weight"])
        params = opt["option_params"]
        if not isinstance(params, dict):
            raise ValueError(f"'option_params' for {name} must be a dict")

        # resolve model
        model_enum = resolve_model(params)
        params = {**params, "model": model_enum}
        params.pop("model_value", None)

        out.append({
            "name": name,
            "weight": weight,
            "option_params": params,
        })

    return out
def choose_option_by_weight(options: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = [o["weight"] for o in options]
    return random.choices(options, weights=weights, k=1)[0]

def change_model(model: Model):
    """
    Change the active model in the Stable Diffusion API.
    This function sends a request to the API to set the specified model.
    """

    payload = {
        "sd_model_checkpoint": model.value,
    }

    try:
        response = call_api("sdapi/v1/options", **payload)
        logging.info(f"Model changed to {model.value}: {response}")
        time.sleep(sleepy_time)
        return response
    except Exception as e:
        logging.error(f"Failed to change model to {model.value}: {e}")


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode("utf-8")  # Convert dict to JSON bytes
    request = urllib.request.Request(
        f"{forge_url_base}/{api_endpoint}",
        headers={"Content-Type": "application/json"},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode("utf-8"))


def call_txt2img_api(index: int, card: dict, **payload):
    resp = call_api("sdapi/v1/txt2img", **payload)

    images = resp.get("images") or []
    if not images:
        # Some SD backends also return 'error' or 'detail' for failures:
        err_msg = resp.get("error") or resp.get("detail") or "no images in response"
        raise RuntimeError(f"[Card #{index+1}] txt2img returned no images: {err_msg}")

    for i_img, image in enumerate(images):
        save_path = os.path.join(out_dir, f"{card['id']}.png")
        decode_and_save_base64(image, save_path)


# under development
def get_SD_prompt(index, card, ai: OpenAI, sd_model="" ,model_str="gpt-5") -> str:
    """Generate a Stable Diffusion prompt for the given card.
    This function extracts relevant information from the card and fetches a SD prompt from ChatGPT Overlord.
    """

    logging.info(f"[Card #{index+1}] Generating SD prompt for {card['name']}...")
    time.sleep(sleepy_time)

    name = card.get("name", "Unknown Card")
    typeLine = card.get("typeLine", "")
    rarity = card.get("rarity", "")
    manaCost = card.get("manaCost", "")
    oracleText = card.get("oracleText", "")
    flavorText = card.get("flavorText", "")

    card_data = {
        "name": name,
        "typeLine": typeLine,
        "rarity": rarity,
        "manaCost": manaCost,
        "oracleText": oracleText,
        "flavorText": flavorText,
    }

    logging.debug(
        f"[Card #{index+1}] Card data for prompt: {json.dumps(card_data, indent=2)}"
    )

    # Construct the prompt for ChatGPT
    if ai is None:
        ai = OpenAI(api_key=API_KEY)

    prompt = ""

    specialised_prompt_info = ""
    if sd_model == Model.FLUX1DEV_HYPERNF4_FLUX1DEVBNB_FLUX1DEVHYPERNF4.value:
        specialised_prompt_info = "This card is designed for the Flux1Dev HyperNF4 model, which needs a different approach to prompts. While SD and SDXL models need comma separated tags, Flux1Dev HyperNF4 model needs a more plain language verbose and descriptive prompt."
    else:
        specialised_prompt_info = "This card is designed for the Stable Diffusion model, which needs a comma separated tags and keywords prompt. The prompt should be descriptive and concise, focusing on the visual aspects of the card."

    input = [
        {
            "role": "system",
            "content": f"You are an assistant who works as a Magic: The Gathering card designer. You review Mtg cards and compose a prompt that could be fitting for a Stable Diffusion image generator. OUTPUT MUST BE THE STABLE DIFFUSION PROMPT ONLY. DO NOT EXPLAIN THE CARD OR YOUR REASONING IN THE FINAL OUTPUT. The input will be information about a playing card as JSON. When composing the prompt, be mindful of the cards power and rarity and generally avoid the use of game terms like +X/+X or 'cards' and explicitly stating Magic the Gathering illustration or similar, that could confuse stable diffusion or generate watermarks. Regarding your Output format always start with the name of the card in brackets like this: '($name:{max_tag_weight}),'. Restrict your Output to approximately 75 Stable Diffusion Tokens. The output must follow Magic the Gathering color pie and card power balancing principles. {specialised_prompt_info}",
        },
        {"role": "user", "content": json.dumps(card_data)},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = ai.responses.create(model=model_str, input=input, timeout=60)
            if response.status in ["failed", "cancelled"]:
                raise response
            break  # Exit loop if request succeeded
        except Exception as e:
            logging.error(f"[Card #{index+1}] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise  # Re-raise last exception if all retries failed
            time.sleep(retry_delay)  # Wait before retrying

    # Check if the request was successful
    if response.status == "completed":
        logging.debug(
            f"[Card #{index+1}] Response from OpenAI: {response.model_dump()}"
        )
        match model_str:
            case "gpt-5":
                prompt = response.model_dump()["output"][1]["content"][0]["text"]
            case _:
                prompt = ""
        logging.info(
            f"[Card #{index+1}] Overlord says: {prompt[:60]} ... {prompt[-60:]}"
        )  # Log first and last 5 characters of the prompt
        time.sleep(sleepy_time)

    else:
        logging.info(f"[Card #{index+1}] Error:", response.status, response.text)
        time.sleep(sleepy_time)

    return prompt


def getCardImage(index: int, card: dict, payload: dict, image_model: Model):
    """
    Get the image for a card using the specified image model.
    Blocking call that returns after the image is generated.
    """
    # Check current model
    req = urllib.request.Request(f"{forge_url_base}/sdapi/v1/options")
    res = urllib.request.urlopen(req)
    res = json.loads(res.read().decode("utf-8"))
    logging.info(f"[Card #{index+1}] Current model: {res.get('sd_model_checkpoint', 'Unknown')}")
    time.sleep(sleepy_time)

    if res.get("sd_model_checkpoint") != image_model.value:
        logging.info(f"[Card #{index+1}] Changing model to {image_model.value} ...")
        time.sleep(sleepy_time)
        change_model(image_model)
    else:
        logging.info(f"[Card #{index+1}] Model {image_model.value} is already active.")
        time.sleep(sleepy_time)

    # Do the actual generation (blocking)
    call_txt2img_api(index, card, **payload)

def get_special_tags(index: int) -> str:
    """
    Generate a string of special tags based on the special_tags dictionary.
    Each tag is formatted as (tag:weight) and separated by commas.
    """

    i = index

    selected_special_tags = []
    selected_special_tags_weights = []
    all_special_tags = special_tags.keys()
    for tag in all_special_tags:
        if merlinAI_lib.check_mutation(special_tags[tag]["chance"]):
            logging.info(f"[Card #{i+1}] Adding special tag: {tag}")
            time.sleep(sleepy_time)
            selected_special_tags.append(tag)
            weight = special_tags[tag]["weight"]
            if vary_special_tags_weights:
                weight = round(
                    merlinAI_lib.truncated_normal_random(mean=weight, sd=loraStDe,high=max_tag_weight, low=0.0),
                    2,
                )
            selected_special_tags_weights.append(weight)
    if selected_special_tags: # target string (tag1:weight1), (tag2:weight2), ...
        special_tags_str = ", ".join(
            [f"({tag}:{weight})" for tag, weight in zip(selected_special_tags, selected_special_tags_weights)]
        )
        logging.info(f"[Card #{i+1}] Special tags: {special_tags_str}")
        time.sleep(sleepy_time)
    else:
        special_tags_str = ""
    
    return special_tags_str

def generate_images_from_dict(
    cards: list[dict],
    config: dict,  # Add config parameter
    option_change_chance: int = 20,
    on_done: Optional[Callable[[dict], None]] = None,  # <- NEW
) -> str:
    """
    Generate images for a list of cards using Stable Diffusion.
    Calls `on_done(card)` once per card when its image attempt finishes.
    Returns the directory where images were written.
    """
    
    # Extract config values and set module globals for backward compatibility
    global forge_url_base, forge_out, random_lora_weights, apply_lora_chance
    global loraStDe, use_special_tags, vary_special_tags_weights, model_swap_chance
    global max_retries, retry_delay, max_tag_weight, sleepy_time, out_dir, special_tags
    
    # Strict config validation - NO FALLBACKS
    try:
        sd_config = config["SD_config"]
    except KeyError:
        raise ValueError("SD_config section missing from configuration")
    
    # Required configuration keys with strict validation
    required_keys = [
        "forge_url_base", "sd_output_dir", "random_lora_weights", "apply_lora_chance",
        "lora_weight_standard_deviation", "use_special_tags", "varying_special_tags_weight",
        "model_swap_chance", "max_retries", "retry_delay", "max_tag_weight", "sleepy_time",
        "base_output_dir", "image_options"
    ]
    
    # Check for missing required keys
    missing_keys = [key for key in required_keys if key not in sd_config]
    if missing_keys:
        raise ValueError(f"Missing required SD_config keys: {missing_keys}")
    
    # Extract and validate all config values
    forge_url_base = sd_config["forge_url_base"]
    if not isinstance(forge_url_base, str) or not forge_url_base:
        raise ValueError("forge_url_base must be a non-empty string")
    
    forge_out = sd_config["sd_output_dir"]
    if not isinstance(forge_out, str) or not forge_out:
        raise ValueError("sd_output_dir must be a non-empty string")
    
    random_lora_weights = sd_config["random_lora_weights"]
    if not isinstance(random_lora_weights, bool):
        raise ValueError("random_lora_weights must be a boolean")
    
    apply_lora_chance = sd_config["apply_lora_chance"]
    if not isinstance(apply_lora_chance, (int, float)) or not (0 <= apply_lora_chance <= 100):
        raise ValueError("apply_lora_chance must be a number between 0 and 100")
    
    loraStDe = sd_config["lora_weight_standard_deviation"]
    if not isinstance(loraStDe, (int, float)) or loraStDe < 0:
        raise ValueError("lora_weight_standard_deviation must be a non-negative number")
    
    use_special_tags = sd_config["use_special_tags"]
    if not isinstance(use_special_tags, bool):
        raise ValueError("use_special_tags must be a boolean")
    
    vary_special_tags_weights = sd_config["varying_special_tags_weight"]
    if not isinstance(vary_special_tags_weights, bool):
        raise ValueError("varying_special_tags_weight must be a boolean")
    
    model_swap_chance = sd_config["model_swap_chance"]
    if not isinstance(model_swap_chance, (int, float)) or not (0 <= model_swap_chance <= 100):
        raise ValueError("model_swap_chance must be a number between 0 and 100")
    
    max_retries = sd_config["max_retries"]
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")
    
    retry_delay = sd_config["retry_delay"]
    if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
        raise ValueError("retry_delay must be a non-negative number")
    
    max_tag_weight = sd_config["max_tag_weight"]
    if not isinstance(max_tag_weight, (int, float)) or max_tag_weight <= 0:
        raise ValueError("max_tag_weight must be a positive number")
    
    sleepy_time = sd_config["sleepy_time"]
    if not isinstance(sleepy_time, (int, float)) or sleepy_time < 0:
        raise ValueError("sleepy_time must be a non-negative number")
    
    # Special tags validation
    special_tags = sd_config.get("special_tags", {})
    if not isinstance(special_tags, dict):
        raise ValueError("special_tags must be a dictionary")
    
    # Update output directory path
    base_output_dir = sd_config["base_output_dir"]
    if not isinstance(base_output_dir, str) or not base_output_dir:
        raise ValueError("base_output_dir must be a non-empty string")
    
    out_dir = os.path.join(base_output_dir, forge_out)
    os.makedirs(out_dir, exist_ok=True)

    cards_data = cards

    all_options = load_image_options_from_config(config=config)
    selected_opt = choose_option_by_weight(all_options)
    stack_params = selected_opt["option_params"]

    for i, card in enumerate(cards_data):
        logging.info(f"[Card #{i+1}] Generating image for {card['name']} ...")
        time.sleep(sleepy_time)

        if merlinAI_lib.check_mutation(option_change_chance):
            logging.info(f"[Card #{i+1}] Changing image generation options randomly.")
            time.sleep(sleepy_time)
            selected_opt = choose_option_by_weight(all_options)
            stack_params = selected_opt["option_params"]

        option = stack_params.copy()
        model = option.pop("model")

        # Build prompt (unchanged) ...
        ai = None
        prompt = get_SD_prompt(i, card, ai, sd_model=model.value, model_str="gpt-5")
        loras = option.pop("loras", "")
        selected_loras = {}
        # check if loras are enabled
        for lora in loras:
            if merlinAI_lib.check_mutation( apply_lora_chance ):
                logging.info(f"[Card #{i+1}] Adding Lora: {lora}")
                time.sleep(sleepy_time)
                selected_loras[lora] = loras[lora]

        if random_lora_weights:
            for lora in selected_loras:
                selected_loras[lora] = round(random.uniform(0, 0.9), 2)

        selected_loras = " ".join([f"<lora:{name}:{weight}>" for name, weight in selected_loras.items()])
        logging.info(f"[Card #{i+1}] Loras: {selected_loras}")
        time.sleep(sleepy_time)
        option["prompt"] = f"{prompt}, {get_special_tags(i)} {selected_loras}" if use_special_tags else f"{prompt} {selected_loras}"

        if config["SD_config"].get("randomise_negative_prompt", False):
            negative_prompt = option.get("negative_prompt", "") or ""
            if (not negative_prompt) or merlinAI_lib.check_mutation(config["SD_config"].get("chance_no_negative_prompt", 50)):
                logging.info(f"[Card #{i+1}] No negative prompt used.")
                time.sleep(sleepy_time)
                option["negative_prompt"] = ""
            else:
                lst = negative_prompt.split(", ")
                option["negative_prompt"] = ", ".join(random.sample(lst, random.randint(0, len(lst))))

        # --- NEW worker with retries based on config ---
        err = {"e": None}

        def _worker():
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    getCardImage(i, card, option, image_model=model)
                    return  # success -> leave worker
                except Exception as e:
                    last_exc = e
                    logging.warning(
                        f"[Card #{i+1}] Image generation attempt {attempt}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay*attempt)  # Exponential backoff
            # if we got here, all retries failed
            err["e"] = last_exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        with tqdm(total=100, desc=f"Image {i+1}", unit="%", position=0, leave=False, dynamic_ncols=True, bar_format=BAR_FMT) as subbar:
            last = -1
            while t.is_alive():
                pct = _sd_progress_percent()
                if pct != last:
                    subbar.n = pct
                    subbar.refresh()
                    last = pct
                time.sleep(0.5)

            # ensure bar ends at 100%
            subbar.n = 100
            subbar.refresh()

        if err["e"]:
            logging.error(f"[Card #{i+1}] Image generation failed: {err['e']}")

        # Always notify to keep overall bar accurate
        if on_done:
            try:
                on_done(card)
            except Exception as cb_e:
                logging.debug(f"on_done callback raised: {cb_e}")
    
    return out_dir

if __name__ == "__main__":
    # Load configuration using new config system with proper argument parsing
    import argparse
    
    # Handle imports for config_manager
    try:
        from . import config_manager
    except ImportError:
        import config_manager
    
    parser = argparse.ArgumentParser(description="Generate Stable Diffusion images for MTG cards")
    parser.add_argument("config", nargs="?", default="configs/user.yml", 
                       help="Path to configuration file (default: configs/user.yml)")
    
    args = parser.parse_args()
    
    try:
        config = config_manager.load_config(args.config)
    except FileNotFoundError:
        logging.error(f"Config file {args.config} not found")
        sys.exit(1)

    # Extract config filename for input naming
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Config values will be extracted within generate_images_from_dict function
    logging.info("Starting image generation from JSON...")
    sleepy_time = config["SD_config"]["sleepy_time"]
    time.sleep(sleepy_time)
    outdir = config["aitomaton_config"]["output_dir"]
    config_outdir = os.path.join(outdir, config_name)
    cardsjson = os.path.join(config_outdir, f"{config_name}_cards.json")
    
    try:
        with open(cardsjson, "r", encoding="utf-8") as f:
            cards_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Generated cards file not found: {cardsjson}")
        logging.error("Please run square_generator.py first to generate cards")
        exit(1)

    try:
        model_swap_chance = config["SD_config"]["model_swap_chance"]
        output = generate_images_from_dict(cards_data, config, option_change_chance=model_swap_chance)
        logging.info(f"Images generated and saved to {output}")
        time.sleep(sleepy_time)
    except Exception as e:
        logging.error(f"Error during image generation: {e}")

    logging.info("Image generation completed.")
    time.sleep(sleepy_time)
