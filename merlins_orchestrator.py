#!/usr/bin/env python3
"""
================================================================================
 Merlin's Aitomaton - MTG Card Generation Orchestrator
--------------------------------------------------------------------------------
 Main CLI interface to orchestrate the complete MTG card generation pipeline:
 1. Card Generation (square_generator.py)
 2. Magic Set Editor conversion (MTGCG_mse.py) 
 3. Stable Diffusion image generation (imagesSD.py)
--------------------------------------------------------------------------------
 Author  : Merlin Duty-Knez
 Date    : August 20, 2025
================================================================================
"""

import os
import sys
import logging
import argparse
import subprocess
import atexit
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import config_manager  # type: ignore
from scripts.metrics import GenerationMetrics  # type: ignore
from scripts.merlinAI_lib import check_and_normalize_config  # type: ignore


# Setup logging (will be configured based on verbose/silent flags)
def setup_logging(verbose: bool = False, silent: bool = False):
    """Configure logging with 3-tier system: silent (ERROR), verbose (DEBUG), normal (INFO)."""
    if silent:
        # Silent mode: ERROR only
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            force=True
        )
    elif verbose:
        # Verbose mode: DEBUG with timestamps
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True
        )
    else:
        # Normal mode: INFO with clean format
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # Clean format for user messages
            force=True
        )


class MerlinsAitomaton:
    """Main orchestrator class for the MTG card generation pipeline."""

    def __init__(self, config_path: str | None, verbose: bool = False, silent: bool = False):
        """Initialize orchestrator with configuration.

        Auto-resolution if config missing:
          - Scan configs/ for *.yml (exclude DEFAULTSCONFIG.yml)
          - None found → defaults only
          - One found → use it
          - Many found: prompt (interactive) or pick first / env override
        """
        self.project_root = Path(__file__).parent
        self.configs_dir = self.project_root / 'configs'
        self.verbose = verbose
        self.silent = silent
        self.config_path: Optional[str] = self._resolve_config_path(config_path)
        self.defaults_only = self.config_path is None
        self._ephemeral_config: Optional[Path] = None
        if self.defaults_only:
            # Create ephemeral config file used only for normalization then cleaned up
            self.config_path = self._create_ephemeral_defaults_config()
            self._ephemeral_config = Path(self.config_path)
            atexit.register(self._cleanup_ephemeral_config)
        self.config = self._load_config()
        self.scripts_dir = self.project_root / "scripts"

    def _resolve_config_path(self, requested: str | None) -> str | None:
        """Resolve which config file to use or return None for defaults-only."""
        # If explicit path provided and exists -> use it
        if requested and Path(requested).exists():
            return requested
        # List candidate user configs
        candidates = []
        if self.configs_dir.exists():
            candidates = [
                str(p) for p in sorted(self.configs_dir.glob('*.yml'))
                if p.name != 'DEFAULTSCONFIG.yml' and not p.name.startswith('__')
            ]
        # Environment override
        env_choice = os.getenv('MERLIN_DEFAULT_CONFIG')
        if env_choice:
            env_path = Path(env_choice)
            if not env_path.is_file():  # allow bare filename inside configs
                maybe = self.configs_dir / env_choice
                if maybe.is_file():
                    env_path = maybe
            if env_path.is_file():
                return str(env_path)
        # Decision tree
        if not candidates:
            # No user configs present -> defaults only
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Multiple candidates
        interactive = sys.stdin.isatty() and os.getenv('MERLIN_NONINTERACTIVE','0') not in ('1','true','TRUE')
        if not interactive:
            return candidates[0]  # deterministic
        # Prompt user
        logging.info("\nAvailable configuration files:")
        for idx, path in enumerate(candidates, 1):
            logging.info(f"  {idx}. {Path(path).name}")
        while True:
            choice = input(f"Select config [1-{len(candidates)}] or ENTER for 1 (defaults-only type 'd'): ").strip()
            if choice.lower() == 'd':
                return None
            if choice == '':
                return candidates[0]
            if choice.isdigit():
                i = int(choice)
                if 1 <= i <= len(candidates):
                    return candidates[i-1]
            logging.warning("Invalid selection, try again.")

    def _create_ephemeral_defaults_config(self) -> str:
        """Create a minimal ephemeral config file to drive normalization using defaults only."""
        tmp = self.configs_dir / '__defaults_autoload.yml'
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        content = [
            '# Auto-generated minimal config for defaults-only mode',
            'skeleton_params:',
            '  types_mode: normal',
        ]
        tmp.write_text('\n'.join(content), encoding='utf-8')
        return str(tmp)

    def _cleanup_ephemeral_config(self):
        """Remove ephemeral defaults-only config file (best-effort)."""
        try:
            if self._ephemeral_config and self._ephemeral_config.exists():
                self._ephemeral_config.unlink()
        except Exception:
            # Silently ignore cleanup issues
            pass

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            # First do basic config loading
            # If defaults-only flag triggered earlier, pass None to loader to get defaults, but we will still run normalization using ephemeral file
            config = config_manager.load_config(self.config_path if not self.defaults_only else None)
            logging.info(f"✅ Configuration loaded from {self.config_path}")
            
            # Apply normalization to ensure runtime uses same normalized weights as validation
            config_path = Path(self.config_path)
            defaults_path = config_path.parent / "DEFAULTSCONFIG.yml"
            
            if defaults_path.exists():
                # Get normalized config (same as validation but without printing)
                normalized_config = self._validate_config()
                if normalized_config is not None:
                    logging.info("✅ Configuration normalized for runtime use")
                    return normalized_config
                else:
                    logging.warning("⚠️ Configuration normalization failed, using basic config")
            else:
                logging.warning(f"⚠️ DEFAULTSCONFIG.yml not found at: {defaults_path}")
                logging.warning("⚠️ Using basic config loading without normalization")
            
            return config
        except FileNotFoundError:
            if self.defaults_only:
                logging.warning("⚠️ Running with defaults only (no user config file found).")
                return config_manager.load_config(None)
            logging.error(f"❌ Config file not found: {self.config_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"❌ Error loading config: {e}")
            sys.exit(1)
    
    def _validate_config(self) -> Dict[str, Any] | None:
        """Run configuration validation and normalization quietly for runtime use."""
        config_path = Path(self.config_path)
        defaults_path = config_path.parent / "DEFAULTSCONFIG.yml"
        
        if not defaults_path.exists():
            return None
        
        try:
            # Use check_and_normalize_config with silent=True for loading without output
            result = check_and_normalize_config(self.config_path, save=False, verbose=False, silent=True)
            
            return result
                
        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            return None
    
    def _run_config_validation(self, save: bool = False):
        """Run full configuration validation using merlinAI_lib with optional save."""
        config_path = Path(self.config_path)
        defaults_path = config_path.parent / "DEFAULTSCONFIG.yml"
        
        if not defaults_path.exists():
            logging.critical(f"\n❌ CRITICAL ERROR: DEFAULTSCONFIG.yml not found at: {defaults_path}")
            logging.critical("   This file is required for configuration validation and normalization.")
            logging.critical("   Please ensure your config file is in the configs/ directory.")
            raise FileNotFoundError(f"Required DEFAULTSCONFIG.yml not found at {defaults_path}")
        
        try:
            logging.info("\n🔍 RUNNING FULL CONFIGURATION CHECK...")
            logging.info("="*60)
            
            # Run the full configuration check and normalize with save option
            # This will validate, normalize weights, and show detailed analysis
            result = check_and_normalize_config(self.config_path, save=save, verbose=self.verbose)
            
            if result is None:
                logging.critical("\n❌ CRITICAL ERROR: Configuration validation failed!")
                logging.critical("   Please fix the errors above and try again.")
                raise ValueError("Configuration validation failed")
            
            logging.info("="*60)
            if save:
                logging.info("💾 Configuration saved with normalized values")
            else:
                logging.info("📋 Configuration check complete - use --save to write changes")
            logging.info("")  # Add spacing after validation results
                
        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            logging.error("Cannot proceed with invalid configuration!")
            sys.exit(1)
    
    def display_config_summary(self):
        """Display a summary of the current configuration."""
        logging.info("\n" + "="*60)
        logging.info("🔧 CONFIGURATION SUMMARY")
        logging.info("="*60)
        
        # Card generation settings
        aitomaton_config = self.config.get("aitomaton_config", {})  # Updated to use correct key
        logging.info(f"📊 Total Cards: {aitomaton_config.get('total_cards', 'N/A')}")
        logging.info(f"🔀 Concurrency: {aitomaton_config.get('concurrency', 'N/A')}")
        logging.info(f"📁 Output Directory: {aitomaton_config.get('output_dir', 'N/A')}")
        
        # API settings
        api_params = self.config.get("api_params", {})
        logging.info(f"🤖 AI Model: {api_params.get('model', 'N/A')}")
        logging.info(f"🎨 Image Model: {api_params.get('image_model', 'N/A')}")
        logging.info(f"💡 Generate Image Prompts: {api_params.get('generate_image_prompt', 'N/A')}")
        
        # MSE/Image settings
        mse_config = self.config.get("mtgcg_mse_config", {})
        logging.info(f"🖼️ Image Method: {mse_config.get('image_method', 'N/A')}")
        
        # Set information
        set_params = self.config.get("set_params", {})
        logging.info(f"🃏 Set Name: {set_params.get('set', 'N/A')}")
        logging.info(f"🔣 Set Themes: {set_params.get('themes', 'N/A')}")

        logging.info("="*60)
    
    def check_prerequisites(self) -> bool:
        """Check if all required environment variables and files are present."""
        missing = []
        warnings = []

        # Check required environment variables
        required_env_vars = ["MTGCG_USERNAME", "MTGCG_PASSWORD", "API_KEY"]
        for var in required_env_vars:
            if not os.getenv(var):
                missing.append(f"Environment variable: {var}")

        # Check optional environment variables
        optional_env_vars = ["AUTH_TOKEN"]
        for var in optional_env_vars:
            if not os.getenv(var):
                warnings.append(f"Optional environment variable not set: {var}")

        # Check output directory exists or can be created
        output_dir = Path(self.config["aitomaton_config"]["output_dir"])
    
    def ask_user_confirmation(self, question: str, default: bool = True) -> bool:
        """Ask user for yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{question} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        return response in ['y', 'yes', 'true', '1']
    
    def run_square_generator(self, **overrides) -> bool:
        """Run the card generation step using the normalized config."""
        logging.info("\n🎲 RUNNING CARD GENERATION...")
        
        try:
            # Import the generation function
            from scripts.square_generator import generate_cards
            
            # Use the normalized config that's already loaded
            config = self.config.copy()
            
            # Apply CLI overrides to the config
            for key, value in overrides.items():
                if key == "total_cards":
                    config["aitomaton_config"]["total_cards"] = value
                elif key == "concurrency":
                    config["aitomaton_config"]["concurrency"] = value
                elif key == "image_model":
                    config["api_params"]["image_model"] = str(value)

            pack_builder = config["pack_builder"]
            
            if pack_builder["enabled"]:
                total = 0
                pack = pack_builder["pack"]
                for slot in pack:
                    total += slot["count"]
                config["aitomaton_config"]["total_cards"] = total
                logging.info(f"\n ⚠️ Pack builder enabled: generating {total} cards as per pack configuration")

            # Extract config name from the config path
            config_name = Path(self.config_path).stem
            
            # Call the generation function directly with normalized config
            result = generate_cards(config, config_name, verbose=self.verbose, silent=self.silent)
            
            logging.info("✅ Card generation completed successfully!")
            if self.verbose:
                logging.debug(f"Generated {result['metrics']['successful']} cards")
                logging.debug(f"Output saved to: {result['output_file']}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Card generation failed: {e}")
            if self.verbose:
                logging.exception("Full error details:")
            return False
    
    def run_mse_conversion(self) -> bool:
        """Run the Magic Set Editor conversion step (includes image handling)."""
        current_image_method = self.config.get("mtgcg_mse_config", {}).get("image_method", "download")
        logging.info(f"\n📋 RUNNING MSE CONVERSION + IMAGES (method: {current_image_method})...")
        
        try:
            # Import and call MTGCG_mse function directly with orchestrator-processed config
            from scripts.MTGCG_mse import main_with_config
            if self.verbose:
                logging.debug(f"Running MSE conversion with processed config")
            
            # Pass the already processed config directly
            main_with_config(self.config_path, self.config, verbose=self.verbose, silent=self.silent)
            
            logging.info("✅ MSE conversion + image handling completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ MSE conversion failed with exit code {e.returncode}")
            return False
        except Exception as e:
            logging.error(f"❌ Unexpected error during MSE conversion: {e}")
            return False
    
    def run_tts_export(self, export_format: str = "png", output_dir: Optional[str] = None, mode: str = "complete") -> bool:
        """Run the TTS (Tabletop Simulator) export step."""
        if mode == "complete":
            logging.info(f"\n🎮 RUNNING COMPLETE TTS EXPORT...")
        else:
            logging.info(f"\n🖼️ RUNNING TTS IMAGE EXPORT (format: {export_format})...")
        
        try:
            if mode == "complete":
                # Import and call complete TTS export with orchestrator-processed config  
                from scripts.exportToTTS import export_complete_tts_deck
                if self.verbose:
                    logging.debug(f"Running complete TTS export with processed config")
                
                # Pass the already processed config directly
                success = export_complete_tts_deck(
                    config=self.config,
                    config_path=self.config_path,
                    output_dir=output_dir,
                    verbose=self.verbose,
                    silent=self.silent
                )
            else:
                # Import and call image export only 
                from scripts.exportToTTS import export_card_images_with_mse
                if self.verbose:
                    logging.debug(f"Running TTS image export with processed config")
                
                # Pass the already processed config directly
                success = export_card_images_with_mse(
                    config=self.config,
                    config_path=self.config_path,
                    output_format=export_format,
                    output_dir=output_dir,
                    verbose=self.verbose,
                    silent=self.silent
                )
            
            if success:
                if mode == "complete":
                    logging.info("✅ Complete TTS export completed successfully!")
                else:
                    logging.info("✅ TTS image export completed successfully!")
                return True
            else:
                if mode == "complete":
                    logging.error("❌ Complete TTS export failed!")
                else:
                    logging.error("❌ TTS image export failed!")
                return False
            
        except Exception as e:
            logging.error(f"❌ Unexpected error during TTS export: {e}")
            if self.verbose:
                logging.exception("Full error details:")
            return False

    def interactive_mode(self):
        """Run the orchestrator in interactive mode."""
        logging.info("\n🚀 WELCOME TO MERLIN'S AITOMATON - MTG CARD GENERATION ORCHESTRATOR")
        logging.info("="*71)
        
        # Run config validation for interactive mode (no save)
        self._run_config_validation(save=False)
        
        # Display configuration summary
        self.display_config_summary()
        
        # Check prerequisites
        if not self.check_prerequisites():
            if not self.ask_user_confirmation("Continue anyway?", default=False):
                logging.info("Exiting due to prerequisite issues.")
                return
        
        logging.info("\n🎯 PIPELINE STEPS:")
        logging.info("   1. Generate Cards (square_generator.py)")
        logging.info("   2. Convert to MSE + Images (MTGCG_mse.py)")
        logging.info("      └─ Images handled via config 'image_method' setting")
        logging.info("   3. Export for TTS (exportToTTS.py) - Optional")
        logging.info("      └─ Export card images with meaningful names")
        
        # Step 1: Card Generation
        logging.info("\n" + "="*50)
        current_cards = self.config["aitomaton_config"]["total_cards"]
        current_image_model = self.config["api_params"]["image_model"]
        current_concurrency = self.config["aitomaton_config"]["concurrency"]
        
        if self.ask_user_confirmation(
            f"🎲 Generate {current_cards} cards with image model '{current_image_model}' using {current_concurrency} threads?"
        ):
            overrides = {}
            
            # Ask for modifications
            if self.ask_user_confirmation("Modify any settings?", default=False):
                try:
                    new_cards = input(f"Total cards [{current_cards}]: ").strip()
                    if new_cards:
                        overrides["total_cards"] = int(new_cards)
                    
                    new_concurrency = input(f"Concurrency [{current_concurrency}]: ").strip()
                    if new_concurrency:
                        overrides["concurrency"] = int(new_concurrency)
                    
                    new_image_model = input(f"Image model (dall-e-3/dall-e-2/none) [{current_image_model}]: ").strip()
                    if new_image_model:
                        overrides["image_model"] = new_image_model
                        
                except ValueError as e:
                    logging.error(f"Invalid input: {e}")
                    return
            
            if not self.run_square_generator(**overrides):
                if not self.ask_user_confirmation("Continue with remaining steps despite failure?", default=False):
                    logging.error("❌ Stopping pipeline due to card generation failure.")
                    return
        else:
            logging.info("⏭️ Skipping card generation...")
        
        # Step 2: MSE Conversion + Images
        logging.info("\n" + "="*50)
        current_image_method = self.config.get("mtgcg_mse_config", {}).get("image_method", "download")
        if self.ask_user_confirmation(f"📋 Convert cards to MSE format + handle images (method: {current_image_method})?"):
            if not self.run_mse_conversion():
                logging.error("❌ MSE conversion failed.")
                if not self.ask_user_confirmation("Continue with TTS export despite MSE failure?", default=False):
                    logging.error("❌ Stopping pipeline due to MSE conversion failure.")
                    return
        else:
            logging.info("⏭️ Skipping MSE conversion...")
        
        # Step 3: TTS Export (Optional)
        logging.info("\n" + "="*50)
        if self.ask_user_confirmation("🖼️ Export card images for Tabletop Simulator (TTS)?"):
            format_choice = input("Image format (png/jpg/bmp) [png]: ").strip().lower()
            if not format_choice:
                format_choice = "png"
            elif format_choice not in ["png", "jpg", "bmp"]:
                logging.warning(f"⚠️ Unknown format '{format_choice}', using 'png'")
                format_choice = "png"
            
            custom_dir = input("Custom output directory (leave empty for default): ").strip()
            output_dir = custom_dir if custom_dir else None
            
            if not self.run_tts_export(export_format=format_choice, output_dir=output_dir):
                logging.error("❌ TTS export failed.")
        else:
            logging.info("⏭️ Skipping TTS export...")
        
        logging.info("\n🎉 PIPELINE COMPLETE!")
        logging.info("="*30)
        
        # Show final results
        self.show_results()
        
    def show_results(self):
        """Display results summary after completion."""
        output_dir = Path(self.config["aitomaton_config"]["output_dir"])  # Base output directory
        config_name = Path(self.config_path).stem
        config_outdir = output_dir / config_name  # New schema: per-config subdirectory

        logging.info(f"📁 Base output dir : {output_dir.absolute()}")
        logging.info(f"📁 Config output dir: {config_outdir.absolute()} (new schema)")

        cards_file_new = config_outdir / f"{config_name}_cards.json"
        mse_set_file_new = config_outdir / f"{config_name}-mse-out.mse-set"

        # Legacy fallback paths (pre schema change)
        legacy_cards_file = output_dir / "generated_cards.json"
        legacy_mse_file = output_dir / "mse-out.mse-set"

        results: list[str] = []

        if cards_file_new.exists():
            results.append(f"✅ {cards_file_new.relative_to(output_dir)} - Card data")
        elif legacy_cards_file.exists():
            results.append("✅ generated_cards.json - Card data (legacy location)")
        else:
            results.append("❌ Card data file not found")

        if mse_set_file_new.exists():
            results.append(f"✅ {mse_set_file_new.relative_to(output_dir)} - MSE set file")
        elif legacy_mse_file.exists():
            results.append("✅ mse-out.mse-set - MSE set file (legacy location)")
        else:
            results.append("❌ MSE set file not found")

        # The temporary mse-out directory is removed after zipping in new schema, so we don't list PNGs there anymore.
        # If future image pipelines produce a persistent image directory, detection can be added here.

        # Attempt to detect Forge output (may be placed under config subdir or base)
        forge_candidates = [config_outdir / "forge_out", output_dir / "forge_out"]
        for candidate in forge_candidates:
            if candidate.exists():
                results.append(f"✅ {candidate.relative_to(output_dir)} - Forge format files")
                break

        logging.info("\n📊 Generated files:")
        for r in results:
            logging.info(f"   {r}")

        # Helpful next-step hint
        if mse_set_file_new.exists():
            logging.info(f"\n💡 To view your cards, open {mse_set_file_new} in Magic Set Editor")
        elif legacy_mse_file.exists():
            logging.info(f"\n💡 To view your cards, open {legacy_mse_file} in Magic Set Editor (legacy path)")
        else:
            logging.info("\nℹ️  Run the MSE conversion step to create a .mse-set archive.")
    
    def module_mode(self, steps: list):
        """Run the orchestrator in module mode with specified steps."""
        logging.info(f"\n🤖 RUNNING MODULE MODE: {' -> '.join(steps)}")
        
        # Run config validation for module mode (no save)
        self._run_config_validation(save=False)
        
        success = True
        
        if "cards" in steps:
            success &= self.run_square_generator()
        
        if "mse" in steps and success:
            success &= self.run_mse_conversion()
        
        if "tts" in steps and success:
            success &= self.run_tts_export()
        
        # Note: 'images' step is handled within MTGCG_mse.py based on config
        if "images" in steps:
            logging.info("ℹ️  Images are handled automatically by the MSE conversion step")
            logging.info("   Configure 'mtgcg_mse_config.image_method' in your config file")
        
        if success:
            logging.info("\n🎉 MODULE PROCESSING COMPLETE!")
        else:
            logging.error("\n❌ MODULE PROCESSING FAILED!")
            sys.exit(1)

    def batch_mode(self, batch_count: int, steps: Optional[list] = None):
        """Run the specified pipeline steps multiple times with numbered outputs."""
        if steps is None:
            steps = ["cards", "mse", "tts"]  # Default to full pipeline
            
        logging.info(f"\n🔄 RUNNING BATCH MODE: {batch_count} iterations")
        logging.info(f"🎯 Pipeline steps: {' -> '.join(steps)}")
        
        # Run config validation for batch mode (no save) - this is obligatory!
        self._run_config_validation(save=False)
        
        config_name = Path(self.config_path).stem
        base_output_dir = Path(self.config["aitomaton_config"]["output_dir"])
        batch_output_dir = base_output_dir / config_name
        
        logging.info(f"📁 Batch output directory: {batch_output_dir}")
        
        # Ensure output directory exists
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        successes = 0
        failures = 0
        
        for i in range(1, batch_count + 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"🚀 BATCH ITERATION {i}/{batch_count}")
            logging.info(f"{'='*60}")
            
            # Create a deep copy of the config for this iteration
            import copy
            iteration_config = copy.deepcopy(self.config)
            
            # Update output paths - all files go to the same directory, just with numbered names
            iteration_config["aitomaton_config"] = iteration_config["aitomaton_config"].copy()
            iteration_config["aitomaton_config"]["output_dir"] = str(batch_output_dir)
            
            try:
                # Set environment variables for this iteration (non-logging ones)
                os.environ['MERLIN_ORCHESTRATED'] = '1'
                os.environ['MERLIN_BATCH_ITERATION'] = str(i)
                
                # Create a virtual config path for iteration-specific naming
                iteration_config_path = f"configs/{config_name}-{i}.yml"
                
                iteration_success = True
                
                # Run only the specified steps in order
                if "cards" in steps:
                    logging.info(f"\n🎲 RUNNING CARD GENERATION (iteration {i})...")
                    
                    # Import and run card generation
                    from scripts.square_generator import generate_cards
                    
                    # Handle pack builder override
                    pack_builder = iteration_config["pack_builder"]
                    if pack_builder["enabled"]:
                        if "pack" in pack_builder and pack_builder["pack"]:
                            total = sum(slot["count"] for slot in pack_builder["pack"])
                            iteration_config["aitomaton_config"]["total_cards"] = total
                            if i == 1:  # Only show this message once
                                logging.info(f"📦 Pack builder enabled: generating {total} cards per iteration")
                        else:
                            if i == 1:  # Only show this message once
                                logging.info(f"📦 Pack builder enabled but no pack definition found - using total_cards setting")
                    
                    # Generate cards with iteration-specific config
                    result = generate_cards(iteration_config, f"{config_name}-{i}", verbose=self.verbose, silent=self.silent)
                    if not result:
                        iteration_success = False
                        logging.error(f"❌ Card generation failed for iteration {i}")
                
                if "mse" in steps and iteration_success:
                    logging.info(f"\n📋 RUNNING MSE CONVERSION (iteration {i})...")
                    from scripts.MTGCG_mse import main_with_config
                    
                    try:
                        main_with_config(iteration_config_path, iteration_config)
                    except Exception as e:
                        iteration_success = False
                        logging.error(f"❌ MSE conversion failed for iteration {i}: {e}")
                
                if "tts" in steps and iteration_success:
                    logging.info(f"\n🎮 RUNNING COMPLETE TTS EXPORT (iteration {i})...")
                    
                    # Use the normal TTS export directory structure - don't override output_dir
                    # This ensures proper cleanup and consistent file organization
                    from scripts.exportToTTS import export_complete_tts_deck
                    tts_success = export_complete_tts_deck(
                        config=iteration_config,
                        config_path=iteration_config_path  # Use iteration-specific config path
                    )
                    
                    if not tts_success:
                        iteration_success = False
                        logging.error(f"❌ Complete TTS export failed for iteration {i}")
                
                # Note: 'images' step is handled within MTGCG_mse.py based on config
                if "images" in steps:
                    if i == 1:  # Only show this message once
                        logging.info("ℹ️  Images are handled automatically by the MSE conversion step")
                        logging.info("   Configure 'mtgcg_mse_config.image_method' in your config file")
                
                if iteration_success:
                    successes += 1
                    logging.info(f"✅ Iteration {i} completed successfully!")
                else:
                    failures += 1
                    logging.error(f"❌ Iteration {i} failed!")
                
            except Exception as e:
                failures += 1
                logging.error(f"❌ Iteration {i} failed: {str(e)}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Final summary
        logging.info(f"\n{'='*60}")
        logging.info(f"🎯 BATCH MODE COMPLETE")
        logging.info(f"{'='*60}")
        logging.info(f"✅ Successful iterations: {successes}")
        logging.info(f"❌ Failed iterations: {failures}")
        logging.info(f"📁 Output directory: {batch_output_dir}")
        
        if successes > 0:
            logging.info(f"\n🎉 Generated {successes} sets of cards!")
        if failures > 0:
            logging.warning(f"⚠️  {failures} iterations failed")

    def check_mode(self, save: bool = False):
        """Check configuration and display summary without running any steps."""
        logging.info("\n🔍 CONFIGURATION CHECK MODE")
        logging.info("="*50)
        
        # Run full configuration validation with optional save
        self._run_config_validation(save=save)
        
        # Display configuration summary
        self.display_config_summary()
        
        # Check prerequisites 
        logging.info("\n🔧 PREREQUISITE CHECK:")
        prereq_ok = self.check_prerequisites()
        
        if prereq_ok:
            logging.info("✅ All prerequisites satisfied!")
        else:
            logging.warning("⚠️  Some prerequisites have issues (see above)")
        
        # Check output directory structure
        logging.info("\n📁 OUTPUT DIRECTORY STRUCTURE:")
        output_dir = Path(self.config["aitomaton_config"]["output_dir"])
        config_name = Path(self.config_path).stem
        config_subdir = output_dir / config_name
        
        logging.info(f"   Base output directory: {output_dir}")
        logging.info(f"   Config subdirectory: {config_subdir}")
        logging.info(f"   Cards file would be: {config_subdir / f'{config_name}_cards.json'}")
        logging.info(f"   MSE set file would be: {config_subdir / f'{config_name}-mse-out.mse-set'}")
        
        # Check if output files already exist
        cards_file = config_subdir / f"{config_name}_cards.json"
        mse_file = config_subdir / f"{config_name}-mse-out.mse-set"
        
        logging.info("\n📊 EXISTING OUTPUT FILES:")
        if cards_file.exists():
            logging.info(f"   ✅ Cards file exists: {cards_file}")
        else:
            logging.info(f"   ❌ Cards file not found: {cards_file}")
            
        if mse_file.exists():
            logging.info(f"   ✅ MSE set file exists: {mse_file}")
        else:
            logging.info(f"   ❌ MSE set file not found: {mse_file}")
        
        logging.info(f"\n✅ Configuration check complete for: {self.config_path}")
        logging.info("   Use without --check to run the pipeline.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merlin's Aitomaton - MTG Card Generation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s -y                                 # Run full pipeline (cards + mse + tts)
  %(prog)s --module cards mse tts             # Run all steps including TTS export
  %(prog)s --module cards                     # Only generate cards
  %(prog)s --module tts                       # Only export TTS images (requires existing MSE set)
  %(prog)s my_config.yml -y                   # Use custom config, run full pipeline
  %(prog)s my_config.yml --module mse         # Use custom config, run MSE only
  %(prog)s my_config.yml --batch 5            # Run full pipeline 5 times with numbered outputs
  %(prog)s --batch 3                          # Interactive config selection, then run 3 times
  %(prog)s my_config.yml --batch 5 --module cards    # Generate cards only, 5 times
  %(prog)s my_config.yml --batch 3 --module mse tts  # MSE + TTS export, 3 times
  %(prog)s my_config.yml --check             # Check config without running
  %(prog)s my_config.yml --check --save      # Check config and save normalized values
        """
    )
    
    parser.add_argument(
        "config", 
        nargs="?", 
        default=None,
        help="Path to configuration file. If omitted: auto-select or defaults-only."
    )
    parser.add_argument(
        "--list-configs", action="store_true", help="List available configs and exit"
    )
    
    parser.add_argument(
        "--module", 
        nargs="*",
        choices=["cards", "mse", "images", "tts"],
        help="Run in module mode with specified steps (images handled by mse step, tts exports card images)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        metavar="N",
        help="Run the pipeline N times with numbered outputs (use with --module to specify steps)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--silent", "-s",
        action="store_true",
        help="Enable silent logging (errors only)"
    )
    
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Run full pipeline (equivalent to --module cards mse tts)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration and display summary without running any steps"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save normalized configuration changes when using --check (overwrites config file)"
    )
    
    args = parser.parse_args()
    
    # Handle -y flag by setting modules to full pipeline
    if args.yes:
        if args.module is not None:
            parser.error("-y/--yes cannot be used with --module (use one or the other)")
        if args.check:
            parser.error("-y/--yes cannot be used with --check (use --check alone for validation)")
        args.module = ["cards", "mse", "tts"]
    
    # Validate argument combinations
    if args.save and not args.check:
        parser.error("--save can only be used with --check")
    
    if args.batch and args.check:
        parser.error("--batch and --check cannot be used together")
    
    # Setup logging based on verbose/silent flags
    setup_logging(verbose=args.verbose, silent=args.silent)
    
    # Initialize orchestrator
    if args.list_configs:
        configs_dir = Path(__file__).parent / 'configs'
        if configs_dir.exists():
            files = [p for p in configs_dir.glob('*.yml')
                     if p.name != 'DEFAULTSCONFIG.yml' and not p.name.startswith('__')]
            if not files:
                logging.info("(no user config files found)")
            else:
                for f in sorted(files):
                    logging.info(f.name)
        else:
            logging.info("configs/ directory not found")
        return

    # Handle batch mode - may need interactive config selection
    if args.batch:
        if not args.config:
            # No config specified, need to prompt for one
            logging.info("🔄 BATCH MODE: No config specified, selecting configuration...")
            # Create a temporary orchestrator just to resolve config path
            temp_orchestrator = MerlinsAitomaton(None, verbose=args.verbose, silent=args.silent)
            if temp_orchestrator.defaults_only:
                logging.error("❌ Batch mode requires a specific configuration file")
                logging.error("   Please specify a config file or ensure config files exist in configs/")
                return
            selected_config = temp_orchestrator.config_path
            logging.info(f"✅ Selected configuration: {selected_config}")
        else:
            selected_config = args.config
        
        # Create orchestrator with selected config
        orchestrator = MerlinsAitomaton(selected_config, verbose=args.verbose, silent=args.silent)
        orchestrator.batch_mode(args.batch, args.module)
        return

    orchestrator = MerlinsAitomaton(args.config, verbose=args.verbose, silent=args.silent)
    
    # Run in appropriate mode
    if args.check:
        orchestrator.check_mode(save=args.save)
    elif args.module is not None:
        orchestrator.module_mode(args.module)
    else:
        orchestrator.interactive_mode()


if __name__ == "__main__":
    main()
