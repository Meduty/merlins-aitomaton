#!/usr/bin/env python3
"""
================================================================================
 Export to Tabletop Simulator (TTS) - Complete TTS Deck Export
--------------------------------------------------------------------------------
 Exports card images from MSE set files and creates TTS deck files using
 tts-deckconverter utility for use in Tabletop Simulator.
 
 Features robust upload handling with:
 - Configurable timeout and retry settings from http_config
 - Exponential backoff retry logic for failed uploads
 - Comprehensive error handling for network issues
--------------------------------------------------------------------------------
 Author  : Merlin Duty-Knez
 Date    : August 26, 2025
================================================================================
"""

import os
import sys
import json
import logging
import subprocess
import argparse
import re
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from . import config_manager
except ImportError:
    # When running directly (not as a module)
    import config_manager

def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename by removing characters that MSE typically removes.
    This matches how MSE exports card images.
    """
    # Remove common punctuation that MSE removes: commas, quotes, etc.
    sanitized = re.sub(r'[,"\':;!?]', '', name)
    # Replace other problematic characters with spaces or remove them
    sanitized = re.sub(r'[<>:\"/\\|*]', ' ', sanitized)
    # Clean up multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

# Setup logging
def setup_logging(verbose: bool = False):
    """Configure logging based on verbose flag."""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            force=True
        )


def find_mse_set_file(config: Dict[str, Any], config_path: str) -> Optional[Path]:
    """
    Find the MSE set file for a given configuration.
    
    Args:
        config: Configuration dictionary
        config_path: Path to the configuration file
        
    Returns:
        Path to the MSE set file if found, None otherwise
    """
    # Extract config name for file naming
    config_name = Path(config_path).stem
    
    # Get output directory
    output_dir = Path(config["aitomaton_config"]["output_dir"])
    config_output_dir = output_dir / config_name
    
    # Look for MSE set file
    mse_set_file = config_output_dir / f"{config_name}-mse-out.mse-set"
    
    if mse_set_file.exists():
        return mse_set_file
    else:
        logging.error(f"MSE set file not found: {mse_set_file}")
        return None


def export_card_images_with_mse(config: Dict[str, Any], config_path: str, 
                               output_format: str = "png", 
                               output_dir: Optional[str] = None) -> bool:
    """
    Export card images from MSE set file using Magic Set Editor command line.
    
    Args:
        config: Configuration dictionary
        config_path: Path to the configuration file
        output_format: Image format (png, jpg, etc.)
        output_dir: Custom output directory (optional)
        
    Returns:
        True if export successful, False otherwise
    """
    # Get MSE executable path from environment
    mse_exe = os.getenv("MSE_PATH")
    if not mse_exe:
        logging.error("❌ MSE_PATH not set in environment variables")
        logging.error("   Please add MSE_PATH=path/to/mse.exe to your .env file")
        return False
    
    mse_exe_path = Path(mse_exe)
    if not mse_exe_path.exists():
        logging.error(f"❌ MSE executable not found: {mse_exe_path}")
        return False
    
    # Find the MSE set file
    mse_set_file = find_mse_set_file(config, config_path)
    if not mse_set_file:
        return False
    
    # Determine output directory
    if output_dir:
        export_output_dir = Path(output_dir)
    else:
        config_name = Path(config_path).stem
        base_output_dir = Path(config["aitomaton_config"]["output_dir"])
        export_output_dir = base_output_dir / config_name / "exported_images"
    
    # Create output directory
    export_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct MSE command using correct syntax from GUI
    # Format: mse.exe --export-images FILE [IMAGE]
    # IMAGE parameter should be {card.name}.png with literal curly brackets
    filename_pattern = "{card.name}." + output_format
    
    cmd = [
        str(mse_exe_path),
        "--export-images", 
        str(mse_set_file.absolute()),
        filename_pattern
    ]
    
    logging.info(f"🖼️  Exporting card images from: {mse_set_file}")
    logging.info(f"📁 Output directory: {export_output_dir}")
    logging.info(f"🎨 Format: {output_format}")
    logging.info(f"🔧 MSE Command: {' '.join(cmd)}")
    
    try:
        # Run MSE command from the output directory
        # This should make MSE export images to the current working directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid characters instead of crashing
            check=False,  # Don't raise exception on non-zero exit
            cwd=str(export_output_dir)  # Run from output directory so images go there
        )
        
        if result.returncode == 0:
            logging.info("✅ Image export completed successfully!")
            
            # Count exported files
            exported_files = list(export_output_dir.glob(f"*.{output_format}"))
            logging.info(f"📊 Exported {len(exported_files)} card images")
            
            return True
        else:
            logging.error(f"❌ MSE export failed with exit code: {result.returncode}")
            if result.stderr:
                logging.error(f"   Error output: {result.stderr}")
            if result.stdout:
                logging.info(f"   Output: {result.stdout}")
            return False
            
    except FileNotFoundError:
        logging.error(f"❌ Could not execute MSE: {mse_exe_path}")
        return False
    except Exception as e:
        logging.error(f"❌ Error during MSE export: {e}")
        return False


def create_tts_deck_list(config: Dict[str, Any], config_path: str, 
                        image_dir: Path, output_dir: Path) -> Optional[Path]:
    """
    Create a deck list file in the format required by tts-deckconverter.
    
    This function now works independently of cards.json by:
    1. Scanning the MSE export directory for all image files
    2. Validating the count matches expected total_cards from config
    3. Optionally matching with cards.json if available for proper card names
    4. Falling back to image filenames if cards.json is not available
    
    Args:
        config: Configuration dictionary
        config_path: Path to the configuration file
        image_dir: Directory containing exported card images
        output_dir: Directory to save the deck list file
        
    Returns:
        Path to the created deck list file, or None if failed
    """
    config_name = Path(config_path).stem
    expected_total_cards = config["aitomaton_config"]["total_cards"]
    
    # Step 1: Scan the MSE export directory for all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    all_image_files = []
    
    for ext in image_extensions:
        all_image_files.extend(image_dir.glob(ext))
    
    if not all_image_files:
        logging.error(f"❌ No image files found in {image_dir}")
        return None
    
    # Step 2: Validate count matches expected total
    if len(all_image_files) != expected_total_cards:
        logging.error(f"❌ Image count mismatch! Found {len(all_image_files)} images, expected {expected_total_cards}")
        logging.error(f"   Please check MSE export completed properly")
        return None
    
    logging.info(f"✅ Found {len(all_image_files)} images matching expected count of {expected_total_cards}")
    
    # Step 3: Try to load cards.json for proper card names (optional)
    cards_data = None
    cards_json_path = Path(config["aitomaton_config"]["output_dir"]) / config_name / f"{config_name}_cards.json"
    
    if cards_json_path.exists():
        try:
            with open(cards_json_path, 'r', encoding='utf-8') as f:
                cards_data = json.load(f)
            logging.info(f"✅ Loaded cards.json with {len(cards_data)} cards for name matching")
        except Exception as e:
            logging.warning(f"⚠️  Could not load cards.json: {e}")
            logging.warning("   Will use image filenames as card names")
    else:
        logging.info(f"ℹ️  cards.json not found at {cards_json_path}")
        logging.info("   Will use image filenames as card names")
    
    # Step 4: Create deck list entries
    deck_list_lines = []
    
    for image_file in sorted(all_image_files):
        # Get the base filename without extension
        image_name = image_file.stem
        card_name = image_name  # Default: use filename as card name
        
        # Step 5: If we have cards.json, try to match proper card names
        if cards_data:
            # Try to find a matching card by comparing sanitized names
            for card in cards_data:
                card_json_name = card.get("name", "")
                sanitized_card_name = sanitize_filename(card_json_name)
                
                # Match if the sanitized names are the same
                if sanitized_card_name == image_name:
                    card_name = card_json_name  # Use the proper card name from JSON
                    break
        
        # Add to deck list (format: "1 filename.ext (Card Name)")
        deck_list_lines.append(f"1 {image_file.name} ({card_name})")
    
    if not deck_list_lines:
        logging.error("❌ No valid deck list entries created")
        return None
    
    # Step 6: Save deck list file
    deck_list_file = output_dir / f"{config_name}_deck.txt"
    
    try:
        with open(deck_list_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(deck_list_lines))
        
        logging.info(f"📝 Created deck list: {deck_list_file}")
        logging.info(f"📊 Deck contains {len(deck_list_lines)} cards")
        
        if cards_data:
            logging.info("✅ Used proper card names from cards.json")
        else:
            logging.info("ℹ️  Used image filenames as card names")
        
        return deck_list_file
        
    except Exception as e:
        logging.error(f"❌ Failed to save deck list: {e}")
        return None
        
        return deck_list_file
        
    except Exception as e:
        logging.error(f"❌ Failed to create deck list file: {e}")
        return None


def convert_to_tts_deck(deck_list_file: Path, output_dir: Path, 
                       deck_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                       config_path: Optional[str] = None) -> bool:
    """
    Convert deck list to TTS format using tts-deckconverter.
    
    Args:
        deck_list_file: Path to the deck list file
        output_dir: Directory to save the TTS deck file
        deck_name: Name for the deck (optional)
        config: Configuration dictionary (optional)
        config_path: Path to the configuration file (for batch mode support)
        deck_name: Custom deck name (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        True if conversion successful, False otherwise
    """
    # Get tts-deckconverter path from environment or use default
    tts_converter_path = os.getenv("TTS_DECKCONVERTER_PATH", "tts-deckconverter")
    
    # Extract config information for template naming
    config_name = None
    image_format = "png"  # Default format if not specified in config
    if config_path:
        config_name = Path(config_path).stem
        # Handle batch mode virtual config names like "testPack2-1" -> "testPack2"  
        if '-' in config_name and config_name.split('-')[-1].isdigit():
            config_name = '-'.join(config_name.split('-')[:-1])
    else:
        # Fallback to directory structure if config_path not provided
        config_name = Path(output_dir).parent.name
    
    # Get the image format from config
    if config and 'tts_export' in config:
        image_format = config['tts_export'].get('image_format', 'png')
    
    # Check if tts-deckconverter is available
    # If it's just a command name, try to find it in PATH using 'which' or 'where'
    if not os.path.exists(tts_converter_path):
        # Try to find the command in PATH
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['where', tts_converter_path], capture_output=True, text=True)
            else:  # Unix-like
                result = subprocess.run(['which', tts_converter_path], capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"❌ tts-deckconverter not found in PATH: {tts_converter_path}")
                logging.error("   Please install it from: https://github.com/jeandeaual/tts-deckconverter")
                logging.error("   Or set TTS_DECKCONVERTER_PATH to the full executable path in your .env file")
                return False
        except Exception as e:
            logging.error(f"❌ Error checking for tts-deckconverter: {e}")
            logging.error("   Please install it from: https://github.com/jeandeaual/tts-deckconverter")
            logging.error("   Or set TTS_DECKCONVERTER_PATH to the full executable path in your .env file")
            return False
    
    # Set default deck name if not provided
    if not deck_name:
        deck_name = deck_list_file.stem.replace("_deck", "")
    
    # Copy deck list to images directory for tts-deckconverter
    import shutil
    images_dir = output_dir / "images"
    temp_deck_list = images_dir / deck_list_file.name
    shutil.copy2(deck_list_file, temp_deck_list)
    
    # Update command to use the copied deck list
    cmd = [
        tts_converter_path,
        "-mode", "custom",  # Use custom mode for AI-generated cards
        "-template", "manual",  # Create local template
        "-output", "..",  # Output to parent directory
        str(temp_deck_list.name)  # Use just the filename
    ]
    
    logging.info(f"🔧 Converting to TTS format with local template...")
    logging.info(f"📁 Deck list: {temp_deck_list}")
    logging.info(f"📁 Output directory: {output_dir}")
    logging.info(f"🎮 Deck name: {deck_name}")
    logging.info(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(images_dir)  # Run from images directory
        )
        
        if result.returncode == 0:
            logging.info("✅ TTS template conversion completed successfully!")
            
            # Clean up temporary deck list file (only if cleanup enabled)
            cleanup = config.get('tts_export', {}).get('cleanup', True) if config else True
            if cleanup and temp_deck_list.exists():
                temp_deck_list.unlink()
                logging.debug("🧹 Cleaned up temporary deck list file")
            elif temp_deck_list.exists():
                logging.debug(f"🧹 Cleanup disabled - preserving temporary deck list: {temp_deck_list}")
            
            # Rename template files to desired format immediately after creation
            # Support multiple templates for large decks (>70 cards) that tts-deckconverter splits
            template_files = list(output_dir.glob("*Template*.jpg"))
            if template_files:
                logging.info(f"🔍 Found {len(template_files)} template file(s): {[f.name for f in template_files]}")
                
                # Handle single template or multiple templates
                renamed_templates = []
                for i, old_template in enumerate(sorted(template_files, key=lambda x: x.name)):
                    # Determine correct mapping based on tts-deckconverter naming convention
                    # "Template.jpg" = first template (cards 1-70)
                    # "Template 2.jpg" = second template (cards 71-90)
                    # etc.
                    if 'Template 2' in old_template.name:
                        template_number = 2
                    elif 'Template 3' in old_template.name:
                        template_number = 3
                    elif 'Template 4' in old_template.name:
                        template_number = 4
                    elif 'Template.' in old_template.name:
                        template_number = 1
                    else:
                        # Fallback to sequential numbering
                        template_number = i + 1
                    
                    # Use configured image format
                    if len(template_files) == 1:
                        # Single template - use original naming scheme
                        new_template_name = f"{config_name}-decksheet.{image_format}"
                    else:
                        # Multiple templates - use correct template numbers
                        new_template_name = f"{config_name}-decksheet-{template_number}.{image_format}"
                    
                    new_template_path = output_dir / new_template_name
                    
                    logging.info(f"🔄 Attempting to rename: {old_template.name} → {new_template_name}")
                    
                    try:
                        # Remove existing file if it exists to avoid conflicts
                        if new_template_path.exists():
                            new_template_path.unlink()
                            logging.info(f"🗑️  Removed existing file: {new_template_name}")
                        
                        old_template.rename(new_template_path)
                        logging.info(f"✅ Successfully renamed template: {old_template.name} → {new_template_name}")
                        renamed_templates.append((template_number, new_template_path))
                        
                        # Verify the file exists with new name
                        if new_template_path.exists():
                            logging.info(f"✅ Verified new file exists: {new_template_path}")
                        else:
                            logging.error(f"❌ New file does not exist after rename: {new_template_path}")
                            
                    except Exception as e:
                        logging.error(f"❌ Could not rename template file {old_template.name}: {e}")
                
                # Sort templates by their number for correct order
                renamed_templates.sort(key=lambda x: x[0])
                template_files = [path for _, path in renamed_templates]
            else:
                logging.warning("⚠️  No template file found to rename")
            
            # Fix URLs in generated JSON files for TTS compatibility
            _fix_tts_urls(output_dir)
            
            # Reorganize files for portable TTS format
            _organize_for_portable_tts(output_dir, deck_name, config)
            
            # Look for generated TTS files
            tts_files = list(output_dir.glob("*.json"))
            # Look for renamed template files using configured extension
            template_files = list(output_dir.glob(f"{config_name}-decksheet*.{image_format}"))
            
            if tts_files:
                logging.info(f"📊 Generated TTS files: {[f.name for f in tts_files]}")
            
            if template_files:
                logging.info(f"🖼️  Generated template image(s): {[f.name for f in template_files]}")
                
                if len(template_files) == 1:
                    logging.info("📋 MANUAL UPLOAD REQUIRED:")
                    logging.info("   1. Upload the template image to an image hosting service (e.g., Imgur)")
                    logging.info("   2. Edit the .json file and replace the local 'FaceURL' path with the uploaded URL")
                    logging.info("   3. Import the edited .json file into Tabletop Simulator")
                else:
                    logging.info("📋 MANUAL UPLOAD REQUIRED (MULTI-TEMPLATE DECK):")
                    logging.info(f"   1. Upload all {len(template_files)} template images to an image hosting service (e.g., Imgur)")
                    logging.info("   2. Edit the .json file(s) and replace the local 'FaceURL' paths with the uploaded URLs")
                    logging.info("   3. Ensure template URLs match the correct deck objects in TTS")
                    logging.info("   4. Import the edited .json file(s) into Tabletop Simulator")
            
            # Clean up unwanted nested directories created by tts-deckconverter
            import shutil
            nested_output_dir = output_dir / "output"
            if nested_output_dir.exists():
                logging.info("🧹 Cleaning up nested output directories...")
                shutil.rmtree(nested_output_dir)
                logging.info("✅ Cleanup completed")
            
            return True
        else:
            logging.error(f"❌ TTS conversion failed with exit code: {result.returncode}")
            if result.stderr:
                logging.error(f"   Error output: {result.stderr}")
            if result.stdout:
                logging.info(f"   Output: {result.stdout}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error during TTS conversion: {e}")
        return False


def upload_image_to_imgbb(image_path: Path, api_key: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Upload an image to ImgBB with retry logic and return the direct URL.
    
    Args:
        image_path: Path to the image file to upload
        api_key: ImgBB API key
        config: Configuration dictionary with http_config settings
        
    Returns:
        Direct URL to the uploaded image, or None if upload failed
    """
    # Get HTTP config parameters or use defaults
    if config and "http_config" in config:
        http_config = config["http_config"]
        timeout = http_config.get("timeout", 120)
        max_retries = http_config.get("retries", 3)
        base_retry_delay = http_config.get("retry_delay", 30)
    else:
        timeout = 120
        max_retries = 3
        base_retry_delay = 30
    
    for attempt in range(max_retries + 1):
        try:
            logging.debug(f"🔄 Uploading {image_path.name} (attempt {attempt + 1}/{max_retries + 1})")
            
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {'key': api_key}
                
                response = requests.post(
                    'https://api.imgbb.com/1/upload',
                    files=files,
                    data=data,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        # Get the direct URL to the image
                        url = result['data']['url']
                        logging.info(f"✅ Uploaded {image_path.name}")
                        logging.debug(f"📎 URL: {url}")
                        return url
                    else:
                        error_msg = result.get('error', {}).get('message', 'Unknown error')
                        logging.warning(f"⚠️  ImgBB upload failed (attempt {attempt + 1}): {error_msg}")
                        if attempt < max_retries:
                            retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                            logging.info(f"🔄 Retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logging.error(f"❌ ImgBB upload failed after {max_retries + 1} attempts: {error_msg}")
                else:
                    logging.warning(f"⚠️  ImgBB upload failed with status {response.status_code} (attempt {attempt + 1}): {response.text}")
                    if attempt < max_retries:
                        retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                        logging.info(f"🔄 Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logging.error(f"❌ ImgBB upload failed after {max_retries + 1} attempts with status {response.status_code}")
                        
        except requests.exceptions.Timeout:
            logging.warning(f"⚠️  Upload timeout for {image_path.name} (attempt {attempt + 1}) after {timeout}s")
            if attempt < max_retries:
                retry_delay = base_retry_delay * (2 ** attempt)
                logging.info(f"🔄 Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                logging.error(f"❌ Upload failed after {max_retries + 1} timeout attempts")
                
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"⚠️  Connection error uploading {image_path.name} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                retry_delay = base_retry_delay * (2 ** attempt)
                logging.info(f"🔄 Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                logging.error(f"❌ Upload failed after {max_retries + 1} connection attempts")
                
        except Exception as e:
            logging.warning(f"⚠️  Error uploading {image_path.name} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                retry_delay = base_retry_delay * (2 ** attempt)
                logging.info(f"🔄 Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                logging.error(f"❌ Upload failed after {max_retries + 1} attempts: {e}")
    
    return None


def check_tts_environment():
    """
    Check and display TTS environment setup information
    """
    import os
    import platform
    
    logging.info("🔍 Checking TTS environment setup...")
    
    # Check for TTS_SAVEDOBJS_PATH
    tts_path = os.getenv('TTS_SAVEDOBJS_PATH')
    if tts_path:
        tts_path_obj = Path(tts_path)
        if tts_path_obj.exists():
            logging.info(f"✅ TTS_SAVEDOBJS_PATH set and valid: {tts_path}")
            merlin_dir = tts_path_obj / "Merlin's Aitomaton"
            if merlin_dir.exists():
                logging.info(f"✅ Merlin's Aitomaton directory exists: {merlin_dir}")
            else:
                logging.info(f"📁 Merlin's Aitomaton directory will be created: {merlin_dir}")
        else:
            logging.warning(f"⚠️  TTS_SAVEDOBJS_PATH set but directory doesn't exist: {tts_path}")
    else:
        logging.info("❌ TTS_SAVEDOBJS_PATH not set - will use local relative paths")
        
        # Suggest default path based on OS
        if platform.system() == "Windows":
            username = os.getenv('USERNAME', 'YourUsername')
            suggested_path = f"C:/Users/{username}/Documents/My Games/Tabletop Simulator/Saves/Saved Objects"
            logging.info(f"💡 Suggested Windows path: {suggested_path}")
            logging.info("💡 To set: set TTS_SAVEDOBJS_PATH=\"path/to/your/tts/saved/objects\"")
        else:
            logging.info("💡 Please set TTS_SAVEDOBJS_PATH to your TTS Saved Objects folder")
    
    logging.info("")


def export_complete_tts_deck(config: Dict[str, Any], config_path: str, 
                            output_dir: Optional[str] = None,
                            deck_name: Optional[str] = None) -> bool:
    """
    Complete TTS export pipeline: export images, create deck list, and convert to TTS format.
    
    Args:
        config: Configuration dictionary
        config_path: Path to the configuration file
        output_dir: Custom output directory (optional)
        deck_name: Custom deck name (optional)
        
    Returns:
        True if complete export successful, False otherwise
    """
    config_name = Path(config_path).stem
    
    # Check TTS environment setup
    check_tts_environment()
    
    # Get TTS export configuration
    tts_config = config.get('tts_export', {})
    
    # Always use the standard output directory structure for final files
    base_output_dir = Path(config["aitomaton_config"]["output_dir"])
    final_output_dir = base_output_dir / config_name
    
    # Create a temporary working directory for TTS generation
    temp_tts_dir = final_output_dir / "temp_tts_work"
    temp_tts_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output directory - use temp for processing, final for results
    if output_dir:
        export_output_dir = Path(output_dir)
    else:
        export_output_dir = temp_tts_dir
    
    export_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export card images
    logging.info("🖼️  Step 1: Exporting card images...")
    image_dir = export_output_dir / "images"
    
    # Get image format from tts_export config
    image_format = tts_config.get('image_format', 'png')
    
    success = export_card_images_with_mse(
        config=config,
        config_path=config_path,
        output_format=image_format,
        output_dir=str(image_dir)
    )
    
    if not success:
        logging.error("❌ Failed to export card images")
        return False
    
    # Step 2: Create deck list file
    logging.info("📝 Step 2: Creating deck list...")
    deck_list_file = create_tts_deck_list(config, config_path, image_dir, export_output_dir)
    
    if not deck_list_file:
        logging.error("❌ Failed to create deck list")
        return False
    
    # Step 3: Convert to TTS format
    logging.info("🎮 Step 3: Converting to TTS format...")
    success = convert_to_tts_deck(deck_list_file, export_output_dir, deck_name, config, config_path)
    
    if not success:
        logging.error("❌ Failed to convert to TTS format")
        return False
    
    # Step 4: Organize final output files
    logging.info("📁 Step 4: Organizing final output files...")
    _organize_final_output(export_output_dir, final_output_dir, config_name, config)
    
    logging.info("🎉 Complete TTS export pipeline finished successfully!")
    logging.info(f"📁 Final files saved to: {final_output_dir}")
    
    return True


def main_with_config(config_path, config, mode="complete"):
    """
    Main function that can be called from the orchestrator.
    
    Args:
        config_path: Path to configuration file
        config: Pre-loaded configuration dictionary (optional)
        mode: Export mode - "images" for images only, "complete" for full TTS export
    """
    
    if mode == "complete":
        logging.info("🎮 Starting complete TTS export pipeline...")
        success = export_complete_tts_deck(
            config=config,
            config_path=config_path
        )
    else:
        logging.info("🖼️  Starting MSE image export...")
        success = export_card_images_with_mse(
            config=config,
            config_path=config_path,
            output_format="png"
        )
    
    if success:
        logging.info("✅ Export completed successfully!")
    else:
        logging.error("❌ Export failed!")
        
    return success


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Export cards to Tabletop Simulator format"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/test.yml",
        help="Path to configuration file (default: configs/test.yml)"
    )
    parser.add_argument(
        "--mode",
        choices=["images", "complete"],
        default="complete",
        help="Export mode: 'images' for MSE image export only, 'complete' for full TTS pipeline (default: complete)"
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Image format for export (default: png)"
    )
    parser.add_argument(
        "--output-dir",
        help="Custom output directory for exported files"
    )
    parser.add_argument(
        "--deck-name",
        help="Custom deck name for TTS export"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load configuration
        config = config_manager.load_config(args.config)
        logging.info(f"✅ Configuration loaded from {args.config}")
        
        if args.mode == "complete":
            # Complete TTS export pipeline
            success = export_complete_tts_deck(
                config=config,
                config_path=args.config,
                output_dir=args.output_dir,
                deck_name=args.deck_name
            )
        else:
            # Images only
            success = export_card_images_with_mse(
                config=config,
                config_path=args.config,
                output_format=args.format,
                output_dir=args.output_dir
            )
        
        if success:
            logging.info("🎉 Export process completed successfully!")
            sys.exit(0)
        else:
            logging.error("❌ Export process failed")
            sys.exit(1)
            
    except FileNotFoundError as e:
        logging.error(f"❌ Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        sys.exit(1)


def _fix_tts_urls(output_dir: Path):
    """
    Fix file URLs in TTS JSON files to use proper file:// format,
    update filenames to match renamed templates, and add custom cardback URL
    
    Args:
        output_dir: Directory containing generated TTS files
    """
    import json
    
    # Find cardback image
    workspace_dir = Path(__file__).parent.parent  # Go up from scripts/ to workspace root
    cardback_path = workspace_dir / "media" / "MerlinsAitomatonCB.png"
    cardback_url = None
    
    if cardback_path.exists():
        cardback_url = f"file:///{str(cardback_path.absolute()).replace('/', '\\\\')}"
        logging.info(f"🎨 Found cardback image: {cardback_path}")
    else:
        logging.warning(f"⚠️  Cardback not found: {cardback_path}")
    
    # Find renamed template files
    template_files = list(output_dir.glob("*-decksheet*.png")) + list(output_dir.glob("*-decksheet*.jpg"))
    template_map = {}
    
    for template_file in template_files:
        # Extract template number from filename (e.g., testBigDeck3-decksheet-1.png -> 1)
        if '-decksheet-' in template_file.name:
            # Multi-template case
            parts = template_file.stem.split('-decksheet-')
            if len(parts) == 2 and parts[1].isdigit():
                template_num = int(parts[1])
                template_map[template_num] = template_file
        elif template_file.name.endswith('-decksheet.png') or template_file.name.endswith('-decksheet.jpg'):
            # Single template case
            template_map[1] = template_file
    
    logging.info(f"🔍 Found renamed templates: {template_map}")
    
    for json_file in output_dir.glob("*.json"):
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Track if any changes were made
            modified = False
            
            # Fix URLs in the JSON structure
            def fix_urls_recursive(obj, path=""):
                nonlocal modified
                if isinstance(obj, dict):
                    # Check if we're in a CustomDeck structure
                    if 'CustomDeck' in obj:
                        custom_deck = obj['CustomDeck']
                        for deck_id, deck_info in custom_deck.items():
                            if 'FaceURL' in deck_info:
                                try:
                                    deck_num = int(deck_id)
                                    if deck_num in template_map:
                                        # Update to use renamed template file
                                        renamed_template = template_map[deck_num]
                                        new_url = f"file:///{str(renamed_template.absolute()).replace('/', '\\\\')}"
                                        deck_info['FaceURL'] = new_url
                                        modified = True
                                        logging.info(f"🔗 Updated CustomDeck {deck_id} FaceURL to renamed template: {renamed_template.name}")
                                    else:
                                        logging.warning(f"⚠️  No renamed template found for CustomDeck {deck_id}")
                                except ValueError:
                                    logging.warning(f"⚠️  Invalid CustomDeck ID: {deck_id}")
                            
                            # Handle BackURL
                            if 'BackURL' in deck_info:
                                if not deck_info['BackURL'] and cardback_url:
                                    deck_info['BackURL'] = cardback_url
                                    modified = True
                                    logging.info(f"🎨 Added cardback URL to CustomDeck {deck_id}")
                    
                    # Continue with general URL fixing
                    for key, value in obj.items():
                        if key in ['FaceURL', 'BackURL'] and isinstance(value, str):
                            # Handle FaceURL fixes
                            if key == 'FaceURL' and value:
                                # Convert path to proper file:// URL format for TTS
                                if ('\\' in value or '/' in value) and not value.startswith('file:///'):
                                    # Add file:// prefix and ensure Windows backslashes
                                    windows_path = value.replace('/', '\\')
                                    fixed_url = f"file:///{windows_path}"
                                    obj[key] = fixed_url
                                    modified = True
                                    logging.info(f"🔧 Fixed FaceURL in {json_file.name}")
                                elif value.startswith('file:///') and '/' in value[8:]:  # Check path part after file:///
                                    # Convert forward slashes to backslashes in existing file:/// URLs
                                    prefix = 'file:///'
                                    path_part = value[len(prefix):]
                                    windows_path = path_part.replace('/', '\\')
                                    fixed_url = prefix + windows_path
                                    obj[key] = fixed_url
                                    modified = True
                                    logging.info(f"🔧 Fixed FaceURL in {json_file.name}")
                            
                            # Handle BackURL - set cardback if empty and cardback available, or fix existing URLs
                            elif key == 'BackURL':
                                if not value and cardback_url:
                                    # Set cardback if BackURL is empty
                                    obj[key] = cardback_url
                                    modified = True
                                    logging.info(f"🎨 Added cardback URL in {json_file.name}")
                                elif value and value.startswith('file:///') and '/' in value[8:]:
                                    # Fix existing BackURL to use backslashes
                                    prefix = 'file:///'
                                    path_part = value[len(prefix):]
                                    windows_path = path_part.replace('/', '\\')
                                    fixed_url = prefix + windows_path
                                    obj[key] = fixed_url
                                    modified = True
                                    logging.info(f"🔧 Fixed BackURL in {json_file.name}")
                        else:
                            fix_urls_recursive(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        fix_urls_recursive(item, f"{path}[{i}]")
            
            # Apply the fix
            fix_urls_recursive(data)
            
            # Write back if modified
            if modified:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logging.info(f"✅ Fixed URLs in {json_file.name}")
            
        except Exception as e:
            logging.warning(f"⚠️  Could not fix URLs in {json_file.name}: {e}")


def _organize_final_output(temp_dir: Path, final_dir: Path, config_name: str, config: Dict[str, Any]):
    """
    Organize final output files into clean structure:
    - Copy TTS JSON to final location as {config_name}-tts-obj.json
    - Copy to TTS Saved Objects if copy_to_savedobj is enabled
    - Clean up temporary files
    
    Args:
        temp_dir: Temporary working directory
        final_dir: Final output directory (output/configName/)
        config_name: Configuration name for file prefixes
        config: Configuration dictionary
    """
    import shutil
    import json
    
    tts_config = config.get('tts_export', {})
    copy_to_savedobj = tts_config.get('copy_to_savedobj', True)
    
    # Find the generated TTS JSON file
    json_files = list(temp_dir.glob("*.json"))
    if not json_files:
        logging.warning("⚠️  No TTS JSON file found to organize")
        return
    
    tts_json_file = json_files[0]
    
    # Copy to final location with standardized naming
    final_tts_file = final_dir / f"{config_name}-tts-obj.json"
    shutil.copy2(tts_json_file, final_tts_file)
    logging.info(f"✅ Created final TTS object: {final_tts_file.name}")
    
    # Copy to TTS Saved Objects if enabled
    if copy_to_savedobj:
        _copy_to_tts_saved_objects(final_tts_file, config_name, config)
    else:
        logging.info("📋 TTS Saved Objects copy disabled in config")
    
    # Clean up temporary directory only if cleanup is enabled
    cleanup = tts_config.get('cleanup', True)
    if cleanup:
        try:
            shutil.rmtree(temp_dir)
            logging.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logging.warning(f"⚠️  Could not clean up temp directory: {e}")
    else:
        logging.info(f"🧹 Cleanup disabled - temporary files preserved in: {temp_dir}")


def _copy_to_tts_saved_objects(tts_file: Path, deck_name: str, config: Dict[str, Any]):
    """
    Copy TTS object file to TTS Saved Objects directory if TTS_SAVEDOBJS_PATH is set
    Also copy image files if using local upload mode
    
    Args:
        tts_file: Path to the TTS JSON file
        deck_name: Name of the deck
        config: Configuration dictionary
    """
    import shutil
    
    tts_path = os.getenv('TTS_SAVEDOBJS_PATH')
    if not tts_path:
        logging.warning("⚠️  TTS_SAVEDOBJS_PATH not set - cannot copy to TTS Saved Objects")
        logging.info("💡 Set TTS_SAVEDOBJS_PATH in .env to enable automatic TTS integration")
        return
    
    tts_base_path = Path(tts_path)
    if not tts_base_path.exists():
        logging.error(f"❌ TTS_SAVEDOBJS_PATH does not exist: {tts_base_path}")
        logging.info("💡 Please check your TTS_SAVEDOBJS_PATH environment variable")
        return
    
    # Create Merlin's Aitomaton directory
    merlin_dir = tts_base_path / "Merlin's Aitomaton"
    merlin_dir.mkdir(exist_ok=True)
    
    # Copy TTS JSON file
    dest_file = merlin_dir / f"{deck_name}.json"
    shutil.copy2(tts_file, dest_file)
    logging.info(f"📁 Copied to TTS Saved Objects: {dest_file}")
    
    # If using local mode, also copy image files
    tts_config = config.get('tts_export', {})
    upload_mode = tts_config.get('upload_mode', 'imgbb')
    
    if upload_mode == "local":
        logging.info("📁 Local mode: copying image files to TTS Saved Objects...")
        _copy_images_to_tts_saved_objects(merlin_dir, deck_name, config)
    
    logging.info("🎮 Ready to load in Tabletop Simulator!")


def _copy_images_to_tts_saved_objects(merlin_dir: Path, deck_name: str, config: Dict[str, Any]):
    """
    Copy image files to TTS Saved Objects and update JSON URLs for local mode
    
    Args:
        merlin_dir: Merlin's Aitomaton directory in TTS Saved Objects
        deck_name: Name of the deck
        config: Configuration dictionary
    """
    import shutil
    import json
    
    # Create deck_sheets directory
    deck_sheets_dir = merlin_dir / "deck_sheets"
    deck_sheets_dir.mkdir(exist_ok=True)
    
    # Copy cardback image
    workspace_dir = Path(__file__).parent.parent
    cardback_path = workspace_dir / "media" / "MerlinsAitomatonCB.png"
    if cardback_path.exists():
        cardback_dest = deck_sheets_dir / "cardback.png"
        shutil.copy2(cardback_path, cardback_dest)
        logging.info(f"📁 Copied cardback: {cardback_dest.name}")
    
    # Update JSON to use local file URLs
    json_file = merlin_dir / f"{deck_name}.json"
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find template file and copy it
            template_dest = None
            if 'ObjectStates' in data and data['ObjectStates']:
                face_url = None
                for obj_state in data['ObjectStates']:
                    if 'CustomDeck' in obj_state:
                        for deck_id, deck_info in obj_state['CustomDeck'].items():
                            if 'FaceURL' in deck_info:
                                face_url = deck_info['FaceURL']
                                break
                        if face_url:
                            break
                
                # If we found a web URL, we need to find the original template
                if face_url and face_url.startswith('http'):
                    # For local mode, we need to update URLs to point to local files
                    template_dest = deck_sheets_dir / f"{deck_name}.png"
                    
                    # Copy template from workspace
                    workspace_cardback = workspace_dir / "media" / "MerlinsAitomatonCB.png"
                    if workspace_cardback.exists():
                        shutil.copy2(workspace_cardback, template_dest)
                        logging.info(f"📁 Copied template as: {template_dest.name}")
            
            # Update URLs to point to local files
            def update_to_local_urls(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'FaceURL' and isinstance(value, str) and template_dest:
                            abs_path = str(template_dest.absolute()).replace('/', '\\')
                            obj[key] = f"file:///{abs_path}"
                        elif key == 'BackURL' and isinstance(value, str):
                            cardback_file = deck_sheets_dir / "cardback.png"
                            abs_path = str(cardback_file.absolute()).replace('/', '\\')
                            obj[key] = f"file:///{abs_path}"
                        else:
                            update_to_local_urls(value)
                elif isinstance(obj, list):
                    for item in obj:
                        update_to_local_urls(item)
            
            update_to_local_urls(data)
            
            # Write back the updated JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"✅ Updated JSON URLs for local file access")
            
        except Exception as e:
            logging.error(f"❌ Could not update JSON for local mode: {e}")


def _organize_for_portable_tts(output_dir: Path, deck_name: str, config: Dict[str, Any]):
    """
    Organize TTS files based on upload_mode configuration:
    - upload_mode="imgbb": Upload images to ImgBB and use web URLs
    - upload_mode="local": Copy to TTS_SAVEDOBJS_PATH with absolute file URLs
    
    Args:
        output_dir: Directory containing TTS files
        deck_name: Name of the deck
        config: Configuration dictionary containing tts_export settings
    """
    # Get TTS export configuration
    tts_config = config.get('tts_export', {})
    upload_mode = tts_config.get('upload_mode', 'imgbb')
    
    logging.info(f"🔧 TTS Upload Mode: {upload_mode}")
    
    if upload_mode == "imgbb":
        _organize_with_imgbb_upload(output_dir, deck_name, config)
    elif upload_mode == "local":
        _organize_with_local_files(output_dir, deck_name, config)
    else:
        logging.warning(f"⚠️  Unknown upload_mode '{upload_mode}', defaulting to imgbb")
        _organize_with_imgbb_upload(output_dir, deck_name, config)


def _organize_with_imgbb_upload(output_dir: Path, deck_name: str, config: Dict[str, Any]):
    """
    Upload images to ImgBB and update JSON to use web URLs
    """
    import json
    
    # Get image format from config
    tts_config = config.get('tts_export', {})
    image_format = tts_config.get('image_format', 'png')
    
    # Get ImgBB API key
    imgbb_key = os.getenv('IMGBB_KEY')
    if not imgbb_key:
        logging.error("❌ IMGBB_KEY not found in environment variables")
        logging.info("💡 Please set IMGBB_KEY in your .env file to use imgbb upload mode")
        logging.info("💡 Falling back to local file organization...")
        _organize_with_local_files(output_dir, deck_name, config)
        return
    
    logging.info("🌐 Uploading images to ImgBB...")
    
    # Find template files using configured image format
    template_files = list(output_dir.glob(f"*-decksheet*.{image_format}"))
    if not template_files:
        # Fallback to old naming if new naming not found
        template_files = list(output_dir.glob("*Template*.jpg"))
    
    # Upload all template files
    template_urls = []
    if template_files:
        logging.info(f"🌐 Uploading {len(template_files)} template file(s) to ImgBB...")
        for i, template_file in enumerate(sorted(template_files)):
            logging.debug(f"📤 Uploading template {i+1}/{len(template_files)}: {template_file.name}")
            face_url = upload_image_to_imgbb(template_file, imgbb_key, config)
            if face_url:
                template_urls.append(face_url)
                logging.debug(f"✅ Template {i+1} uploaded successfully")
            else:
                logging.error(f"❌ Failed to upload template {i+1}: {template_file.name}")
                return
    else:
        logging.error("❌ No template files found")
        return
    
    if len(template_urls) != len(template_files):
        logging.error(f"❌ Upload mismatch: {len(template_files)} files, {len(template_urls)} successful uploads")
        return
    
    # Upload cardback
    workspace_dir = Path(__file__).parent.parent
    cardback_path = workspace_dir / "media" / "MerlinsAitomatonCB.png"
    back_url = None
    if cardback_path.exists():
        back_url = upload_image_to_imgbb(cardback_path, imgbb_key, config)
        if not back_url:
            logging.error("❌ Failed to upload cardback image")
            return
    else:
        logging.error(f"❌ Cardback not found: {cardback_path}")
        return
    
    # All uploads completed successfully
    logging.info(f"✅ Successfully uploaded {len(template_urls)} template(s) + 1 cardback to ImgBB")
    
    # Update JSON files with web URLs
    json_files = list(output_dir.glob("*.json"))
    if json_files:
        logging.info(f"📝 Updating {len(json_files)} JSON file(s) with uploaded URLs...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                def update_to_web_urls_smart(obj, inside_custom_deck=False):
                    if isinstance(obj, dict):
                        # Look for CustomDeck structure and update FaceURLs correctly
                        if 'CustomDeck' in obj and not inside_custom_deck:
                            custom_deck = obj['CustomDeck']
                            for deck_id, deck_info in custom_deck.items():
                                if 'FaceURL' in deck_info:
                                    # Map deck IDs to correct template URLs
                                    # CustomDeck "1" = first 70 cards = template 1
                                    # CustomDeck "2" = next 20 cards = template 2, etc.
                                    try:
                                        deck_num = int(deck_id)
                                        if deck_num <= len(template_urls):
                                            # Use correct template order: deck 1 -> template 1, deck 2 -> template 2
                                            template_url = template_urls[deck_num - 1]
                                            deck_info['FaceURL'] = template_url
                                            logging.info(f"🔗 Updated CustomDeck {deck_id} FaceURL: {template_url}")
                                        else:
                                            logging.warning(f"⚠️  No template URL available for CustomDeck {deck_id}")
                                    except (ValueError, IndexError) as e:
                                        logging.warning(f"⚠️  Could not map CustomDeck {deck_id}: {e}")
                                
                                if 'BackURL' in deck_info and back_url:
                                    deck_info['BackURL'] = back_url
                                    logging.info(f"🔗 Updated CustomDeck {deck_id} BackURL")
                        
                        # Always recurse through ALL objects to handle nested CustomDecks
                        for key, value in obj.items():
                            if key == 'CustomDeck':
                                # Recurse into CustomDeck with flag set
                                update_to_web_urls_smart(value, inside_custom_deck=True)
                            elif key == 'FaceURL' and isinstance(value, str) and template_urls and not inside_custom_deck:
                                # Fallback: use first template URL only if not already inside a CustomDeck object
                                obj[key] = template_urls[0]
                                logging.info(f"🔗 Updated FaceURL (fallback): {template_urls[0]}")
                            elif key == 'BackURL' and isinstance(value, str) and back_url and not inside_custom_deck:
                                obj[key] = back_url
                                logging.info(f"🔗 Updated BackURL: {back_url}")
                            else:
                                update_to_web_urls_smart(value, inside_custom_deck)
                    elif isinstance(obj, list):
                        for item in obj:
                            update_to_web_urls_smart(item, inside_custom_deck)
                
                update_to_web_urls_smart(data)
                
                # Write back the updated JSON
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logging.info(f"✅ Updated {json_file.name} with web URLs")
                
            except Exception as e:
                logging.error(f"❌ Failed to update {json_file.name}: {e}")
        
        logging.info("🎉 TTS deck ready with online images!")
        logging.info(f"📋 Deck files: {[f.name for f in json_files]}")
        logging.info("🎮 Import these JSON files directly into Tabletop Simulator!")
    else:
        logging.error("❌ No JSON files found to update")


def _organize_with_local_files(output_dir: Path, deck_name: str, config: Dict[str, Any]):
    """
    Copy files to TTS Saved Objects directory with absolute file URLs
    """
    import json
    import shutil
    import os
    
    # Get image format from config
    tts_config = config.get('tts_export', {})
    image_format = tts_config.get('image_format', 'png')
    
    # Get TTS Saved Objects path from environment
    tts_path = os.getenv('TTS_SAVEDOBJS_PATH')
    if not tts_path:
        logging.warning("⚠️  TTS_SAVEDOBJS_PATH not set. Using local relative paths.")
        logging.info("💡 To auto-copy to TTS, set TTS_SAVEDOBJS_PATH environment variable to:")
        logging.info("   Windows: %USERPROFILE%/Documents/My Games/Tabletop Simulator/Saves/Saved Objects")
        logging.info("   Example: TTS_SAVEDOBJS_PATH=\"C:/Users/YourName/Documents/My Games/Tabletop Simulator/Saves/Saved Objects\"")
        _organize_local_relative(output_dir, deck_name, config)
        return
    
    tts_base_path = Path(tts_path)
    if not tts_base_path.exists():
        logging.error(f"❌ TTS_SAVEDOBJS_PATH does not exist: {tts_base_path}")
        logging.info("💡 Please check your TTS_SAVEDOBJS_PATH environment variable")
        _organize_local_relative(output_dir, deck_name, config)
        return
    
    # Create Merlin's Aitomaton directory in TTS Saved Objects
    merlin_dir = tts_base_path / "Merlin's Aitomaton"
    deck_sheets_dir = merlin_dir / "deck_sheets"
    deck_sheets_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"📁 TTS Saved Objects path: {tts_base_path}")
    logging.info(f"📁 Merlin's Aitomaton directory: {merlin_dir}")
    
    # Find and copy template files using configured image format
    template_files = list(output_dir.glob(f"*-decksheet*.{image_format}"))
    if not template_files:
        # Fallback to old naming if new naming not found
        template_files = list(output_dir.glob("*Template*.jpg"))
    
    template_dests = []
    if template_files:
        logging.info(f"📁 Copying {len(template_files)} template file(s) to TTS...")
        for i, template_file in enumerate(sorted(template_files)):
            template_dest = deck_sheets_dir / template_file.name  # Keep the new name
            
            try:
                shutil.copy2(template_file, template_dest)
                template_dests.append(template_dest)
                logging.info(f"📁 Copied template {i+1}/{len(template_files)}: {template_dest.name}")
            except Exception as e:
                logging.error(f"❌ Could not copy template {template_file.name}: {e}")
                return
    else:
        logging.error("❌ No template file found")
        return
    
    # Copy cardback to TTS location
    workspace_dir = Path(__file__).parent.parent
    cardback_path = workspace_dir / "media" / "MerlinsAitomatonCB.png"
    cardback_dest = None
    if cardback_path.exists():
        cardback_dest = deck_sheets_dir / "cardback.png"
        shutil.copy2(cardback_path, cardback_dest)
        logging.info(f"📁 Copied cardback: {cardback_dest}")
    
    # Copy and update JSON files
    json_files = list(output_dir.glob("*.json"))
    if json_files:
        logging.info(f"📝 Processing {len(json_files)} JSON file(s)...")
        
        for json_file in json_files:
            # Create output filename
            if len(json_files) == 1:
                json_dest = merlin_dir / f"{deck_name}.json"
            else:
                json_dest = merlin_dir / f"{deck_name}_{json_file.stem}.json"
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Track which template to use for each deck object
                template_index = 0
                
                # Update URLs to point to TTS Saved Objects absolute paths
                def update_to_tts_paths(obj, inside_custom_deck=False):
                    nonlocal template_index
                    if isinstance(obj, dict):
                        # Look for CustomDeck structure and update FaceURLs correctly
                        if 'CustomDeck' in obj and not inside_custom_deck:
                            custom_deck = obj['CustomDeck']
                            for deck_id, deck_info in custom_deck.items():
                                if 'FaceURL' in deck_info and template_dests:
                                    # Map deck IDs to correct template files
                                    try:
                                        deck_num = int(deck_id)
                                        if deck_num <= len(template_dests):
                                            current_template = template_dests[deck_num - 1]
                                            abs_path = str(current_template.absolute()).replace('/', '\\')
                                            deck_info['FaceURL'] = f"file:///{abs_path}"
                                            logging.info(f"🔗 Updated CustomDeck {deck_id} FaceURL: {deck_info['FaceURL']}")
                                        else:
                                            logging.warning(f"⚠️  No template file available for CustomDeck {deck_id}")
                                    except (ValueError, IndexError) as e:
                                        logging.warning(f"⚠️  Could not map CustomDeck {deck_id}: {e}")
                                
                                if 'BackURL' in deck_info and cardback_dest:
                                    abs_path = str(cardback_dest.absolute()).replace('/', '\\')
                                    deck_info['BackURL'] = f"file:///{abs_path}"
                                    logging.info(f"🔗 Updated CustomDeck {deck_id} BackURL")
                        
                        # Always recurse through ALL objects to handle nested CustomDecks
                        for key, value in obj.items():
                            if key == 'CustomDeck':
                                # Recurse into CustomDeck with flag set
                                update_to_tts_paths(value, inside_custom_deck=True)
                            elif key == 'FaceURL' and isinstance(value, str) and template_dests and not inside_custom_deck:
                                # Fallback: use first template only if not already inside a CustomDeck object
                                current_template = template_dests[0]
                                abs_path = str(current_template.absolute()).replace('/', '\\')
                                obj[key] = f"file:///{abs_path}"
                                logging.info(f"🔗 Updated FaceURL (fallback): {obj[key]}")
                            elif key == 'BackURL' and isinstance(value, str) and cardback_dest and not inside_custom_deck:
                                abs_path = str(cardback_dest.absolute()).replace('/', '\\')
                                obj[key] = f"file:///{abs_path}"
                                logging.info(f"🔗 Updated BackURL: {obj[key]}")
                            else:
                                update_to_tts_paths(value, inside_custom_deck)
                    elif isinstance(obj, list):
                        for item in obj:
                            update_to_tts_paths(item, inside_custom_deck)
                
                update_to_tts_paths(data)
                
                # Write to TTS Saved Objects location
                with open(json_dest, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logging.info(f"✅ Created TTS deck: {json_dest.name}")
                
            except Exception as e:
                logging.error(f"❌ Could not create TTS deck file from {json_file.name}: {e}")
                return
    
    logging.info("🎉 TTS files successfully copied to Saved Objects!")
    logging.info(f"📍 Deck location: {merlin_dir}")
    logging.info("🎮 The deck is now ready to load in Tabletop Simulator!")


def _organize_local_relative(output_dir: Path, deck_name: str, config: Dict[str, Any]):
    """
    Fallback: Organize files locally with relative paths when TTS_SAVEDOBJS_PATH is not set
    """
    import json
    import shutil
    
    # Get image format from config
    tts_config = config.get('tts_export', {})
    image_format = tts_config.get('image_format', 'png')
    
    # Create local deck_sheets directory
    deck_sheets_dir = output_dir / "deck_sheets"
    deck_sheets_dir.mkdir(exist_ok=True)
    
    # Find and copy template files using configured image format
    template_files = list(output_dir.glob(f"*-decksheet*.{image_format}"))
    if not template_files:
        # Fallback to old naming if new naming not found
        template_files = list(output_dir.glob("*Template*.jpg"))
    
    new_templates = []
    if template_files:
        logging.info(f"📁 Moving {len(template_files)} template file(s) locally...")
        for i, template_file in enumerate(sorted(template_files)):
            if len(template_files) == 1:
                new_template = deck_sheets_dir / f"{deck_name}.png"
            else:
                new_template = deck_sheets_dir / f"{deck_name}_template_{i+1}.png"
            
            try:
                shutil.copy2(template_file, new_template)
                template_file.unlink()  # Remove original
                new_templates.append(new_template)
                logging.info(f"📁 Moved template {i+1}/{len(template_files)}: {new_template.name}")
            except Exception as e:
                logging.warning(f"⚠️  Could not move template {template_file.name}: {e}")
                return
    
    # Copy cardback locally
    workspace_dir = Path(__file__).parent.parent
    cardback_path = workspace_dir / "media" / "MerlinsAitomatonCB.png"
    if cardback_path.exists():
        portable_cardback = deck_sheets_dir / "cardback.png"
        shutil.copy2(cardback_path, portable_cardback)
        logging.info(f"� Copied cardback: {portable_cardback.name}")
    
    # Update JSON files to use relative paths
    template_index = 0
    for json_file in output_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            def update_to_relative_paths(obj, inside_custom_deck=False):
                if isinstance(obj, dict):
                    # Look for CustomDeck structure and update FaceURLs correctly
                    if 'CustomDeck' in obj and not inside_custom_deck:
                        custom_deck = obj['CustomDeck']
                        for deck_id, deck_info in custom_deck.items():
                            if 'FaceURL' in deck_info and new_templates:
                                # Map deck IDs to correct template files
                                try:
                                    deck_num = int(deck_id)
                                    if deck_num <= len(new_templates):
                                        current_template = new_templates[deck_num - 1]
                                        deck_info['FaceURL'] = f"deck_sheets/{current_template.name}"
                                        logging.info(f"🔗 Updated CustomDeck {deck_id} FaceURL: {deck_info['FaceURL']}")
                                    else:
                                        logging.warning(f"⚠️  No template file available for CustomDeck {deck_id}")
                                except (ValueError, IndexError) as e:
                                    logging.warning(f"⚠️  Could not map CustomDeck {deck_id}: {e}")
                            
                            if 'BackURL' in deck_info:
                                deck_info['BackURL'] = "deck_sheets/cardback.png"
                                logging.info(f"🔗 Updated CustomDeck {deck_id} BackURL")
                    
                    # Always recurse through ALL objects to handle nested CustomDecks
                    for key, value in obj.items():
                        if key == 'CustomDeck':
                            # Recurse into CustomDeck with flag set
                            update_to_relative_paths(value, inside_custom_deck=True)
                        elif key == 'FaceURL' and isinstance(value, str) and new_templates and not inside_custom_deck:
                            # Fallback: use first template only if not already inside a CustomDeck object
                            obj[key] = f"deck_sheets/{new_templates[0].name}"
                            logging.info(f"🔗 Updated FaceURL (fallback): {obj[key]}")
                        elif key == 'BackURL' and isinstance(value, str) and not inside_custom_deck:
                            obj[key] = "deck_sheets/cardback.png"
                            logging.info(f"🔗 Updated BackURL: {obj[key]}")
                        else:
                            update_to_relative_paths(value, inside_custom_deck)
                elif isinstance(obj, list):
                    for item in obj:
                        update_to_relative_paths(item, inside_custom_deck)
            
            update_to_relative_paths(data)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"✅ Updated {json_file.name} with relative paths")
            
        except Exception as e:
            logging.warning(f"⚠️  Could not update {json_file.name}: {e}")
    
    logging.info("📦 Local TTS files organized with relative paths")
    logging.info("💡 Set TTS_SAVEDOBJS_PATH to auto-copy to TTS Saved Objects")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Export TTS deck from configuration")
    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load configuration with proper defaults merging
        config = config_manager.load_config(args.config_file)
        
        success = export_complete_tts_deck(config, args.config_file)
        if success:
            logging.info("🎉 TTS export completed successfully!")
        else:
            logging.error("❌ Export process failed")
            sys.exit(1)
            
    except FileNotFoundError as e:
        logging.error(f"❌ Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        sys.exit(1)