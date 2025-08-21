"""
Configuration management module for MTG Card Generator.
Handles loading default config and merging with user config.
"""

import yaml
import argparse
import os
from typing import Dict, Any
from pathlib import Path


def deep_merge_dicts(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary (typically defaults)
        override: Override dictionary (typically user config)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(user_config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration by merging defaults with user config.
    
    Args:
        user_config_path: Path to user configuration file
        
    Returns:
        Merged configuration dictionary
    """
    # Get the script directory to find configs
    script_dir = Path(__file__).parent.parent
    defaults_path = script_dir / "configs" / "DEFAULTSCONFIG.yml"
    
    # Load default configuration
    with open(defaults_path, 'r') as f:
        default_config = yaml.safe_load(f)
    
    if not user_config_path:
        return default_config
    
    # Load user configuration if provided
    if os.path.exists(user_config_path):
        with open(user_config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"User config file not found: {user_config_path}")
    
    # Merge configurations
    merged_config = deep_merge_dicts(default_config, user_config)
    
    return merged_config


def parse_args():
    """
    Parse command line arguments for configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="MTG Card Generator")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/user.yml",
        help="Path to user configuration file (YAML) (default: configs/user.yml)"
    )
    parser.add_argument(
        "--total-cards",
        type=int,
        help="Override total number of cards to generate"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override concurrency level"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Apply command line argument overrides to configuration.
    Returns a copy of the config with overrides applied.
    
    Args:
        config: Base configuration dictionary
        args: Parsed command line arguments
        
    Returns:
        Configuration with CLI overrides applied
    """
    import copy
    result_config = copy.deepcopy(config)
    
    if args.total_cards is not None:
        result_config["square_config"]["total_cards"] = args.total_cards
    
    if args.concurrency is not None:
        result_config["square_config"]["concurrency"] = args.concurrency
        
    if args.output_dir is not None:
        result_config["square_config"]["output_dir"] = args.output_dir
    
    return result_config
