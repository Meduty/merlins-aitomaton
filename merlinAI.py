#!/usr/bin/env python3
"""
================================================================================
 MerlinAI - MTG Card Generation Orchestrator
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
import json
import time
import yaml
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


# Setup logging (will be configured based on verbose flag)
def setup_logging(verbose: bool = False):
    """Configure logging based on verbose flag."""
    if verbose:
        # Verbose mode: show all logs
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True
        )
    else:
        # Quiet mode: suppress logs, only show errors and user messages
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            force=True
        )
        # Also suppress logs from sub-modules
        logging.getLogger().setLevel(logging.ERROR)


class MerlinAIOrchestrator:
    """Main orchestrator class for the MTG card generation pipeline."""

    def __init__(self, config_path: str | None, verbose: bool = False):
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
        print("\nAvailable configuration files:")
        for idx, path in enumerate(candidates, 1):
            print(f"  {idx}. {Path(path).name}")
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
            print("Invalid selection, try again.")

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
            # Use check_and_normalize_config but capture output instead of printing
            # Temporarily redirect stdout to suppress print statements
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = check_and_normalize_config(self.config_path, save=False)
            
            return result
                
        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            return None
    
    def _run_config_validation(self, save: bool = False):
        """Run full configuration validation using merlinAI_lib with optional save."""
        config_path = Path(self.config_path)
        defaults_path = config_path.parent / "DEFAULTSCONFIG.yml"
        
        if not defaults_path.exists():
            print(f"\n❌ CRITICAL ERROR: DEFAULTSCONFIG.yml not found at: {defaults_path}")
            print("   This file is required for configuration validation and normalization.")
            print("   Please ensure your config file is in the configs/ directory.")
            raise FileNotFoundError(f"Required DEFAULTSCONFIG.yml not found at {defaults_path}")
        
        try:
            print("\n🔍 RUNNING FULL CONFIGURATION CHECK...")
            print("="*60)
            
            # Run the full configuration check and normalize with save option
            # This will validate, normalize weights, and show detailed analysis
            result = check_and_normalize_config(self.config_path, save=save)
            
            if result is None:
                print("\n❌ CRITICAL ERROR: Configuration validation failed!")
                print("   Please fix the errors above and try again.")
                raise ValueError("Configuration validation failed")
            
            print("="*60)
            if save:
                print("💾 Configuration saved with normalized values")
            else:
                print("📋 Configuration check complete - use --save to write changes")
            print()  # Add spacing after validation results
                
        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            logging.error("Cannot proceed with invalid configuration!")
            sys.exit(1)
    
    def _get_subprocess_env(self) -> Dict[str, str]:
        """Get environment variables for subprocesses."""
        env = os.environ.copy()
        # Pass verbose flag to subprocesses
        env["MERLIN_VERBOSE"] = "1" if self.verbose else "0"
        return env
    
    def display_config_summary(self):
        """Display a summary of the current configuration."""
        print("\n" + "="*60)
        print("🔧 CONFIGURATION SUMMARY")
        print("="*60)
        
        # Card generation settings
        square_config = self.config.get("square_config", {})
        print(f"📊 Total Cards: {square_config.get('total_cards', 'N/A')}")
        print(f"🔀 Concurrency: {square_config.get('concurrency', 'N/A')}")
        print(f"📁 Output Directory: {square_config.get('output_dir', 'N/A')}")
        
        # API settings
        api_params = self.config.get("api_params", {})
        print(f"🤖 AI Model: {api_params.get('model', 'N/A')}")
        print(f"🎨 Image Model: {api_params.get('image_model', 'N/A')}")
        print(f"💡 Generate Image Prompts: {api_params.get('generate_image_prompt', 'N/A')}")
        
        # MSE/Image settings
        mse_config = self.config.get("mtgcg_mse_config", {})
        print(f"🖼️ Image Method: {mse_config.get('image_method', 'N/A')}")
        
        # Set information
        set_params = self.config.get("set_params", {})
        print(f"🃏 Set Name: {set_params.get('set', 'N/A')}")
        print(f"🔣 Set Themes: {set_params.get('themes', 'N/A')}")

        print("="*60)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("\n🔍 CHECKING PREREQUISITES...")
        
        issues = []
        warnings = []
        
        # Check environment variables
        required_env_vars = ['MTGCG_USERNAME', 'MTGCG_PASSWORD', 'API_KEY']
        for var in required_env_vars:
            if not os.getenv(var):
                issues.append(f"Missing environment variable: {var}")
        
        # Check optional environment variables
        optional_env_vars = ['AUTH_TOKEN']
        for var in optional_env_vars:
            if not os.getenv(var):
                warnings.append(f"Optional environment variable not set: {var} (will attempt to login)")
        
        # Check script files exist
        required_scripts = ['square_generator.py', 'MTGCG_mse.py', 'imagesSD.py']
        for script in required_scripts:
            script_path = self.scripts_dir / script
            if not script_path.exists():
                issues.append(f"Missing script: {script_path}")
        
        # Check output directory
        output_dir = Path(self.config["square_config"]["output_dir"])
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"✅ Created output directory: {output_dir}")
            except Exception as e:
                issues.append(f"Cannot create output directory {output_dir}: {e}")
        
        # Show results
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   • {warning}")
        
        if issues:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ All prerequisites met!")
            return True
    
    def ask_user_confirmation(self, question: str, default: bool = True) -> bool:
        """Ask user for yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{question} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        return response in ['y', 'yes', 'true', '1']
    
    def run_square_generator(self, **overrides) -> bool:
        """Run the card generation step using the normalized config."""
        print("\n🎲 RUNNING CARD GENERATION...")
        
        try:
            # Set environment variable BEFORE importing to control logging
            os.environ['MERLIN_VERBOSE'] = "1" if self.verbose else "0"
            os.environ['MERLIN_ORCHESTRATED'] = '1'
            
            # Import the generation function
            from scripts.square_generator import generate_cards
            
            # Use the normalized config that's already loaded
            config = self.config.copy()
            
            # Apply CLI overrides to the config
            for key, value in overrides.items():
                if key == "total_cards":
                    config["square_config"]["total_cards"] = value
                elif key == "concurrency":
                    config["square_config"]["concurrency"] = value
                elif key == "image_model":
                    config["api_params"]["image_model"] = str(value)
            
            # Extract config name from the config path
            config_name = Path(self.config_path).stem
            
            # Call the generation function directly with normalized config
            result = generate_cards(config, config_name)
            
            print("✅ Card generation completed successfully!")
            if self.verbose:
                logging.info(f"Generated {result['metrics']['successful']} cards")
                logging.info(f"Output saved to: {result['output_file']}")
            return True
            
        except Exception as e:
            print(f"❌ Card generation failed: {e}")
            if self.verbose:
                logging.exception("Full error details:")
            return False
    
    def run_mse_conversion(self) -> bool:
        """Run the Magic Set Editor conversion step (includes image handling)."""
        current_image_method = self.config.get("mtgcg_mse_config", {}).get("image_method", "download")
        print(f"\n📋 RUNNING MSE CONVERSION + IMAGES (method: {current_image_method})...")
        
        try:
            # Import and call MTGCG_mse function directly with orchestrator-processed config
            from scripts.MTGCG_mse import main_with_config
            if self.verbose:
                logging.info(f"Running MSE conversion with processed config")
            
            # Pass the already processed config directly
            main_with_config(self.config_path, self.config)
            
            print("✅ MSE conversion + image handling completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ MSE conversion failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during MSE conversion: {e}")
            return False
    
    def interactive_mode(self):
        """Run the orchestrator in interactive mode."""
        print("\n🚀 WELCOME TO MERLINAI - MTG CARD GENERATION ORCHESTRATOR")
        print("="*65)
        
        # Run config validation for interactive mode (no save)
        self._run_config_validation(save=False)
        
        # Display configuration summary
        self.display_config_summary()
        
        # Check prerequisites
        if not self.check_prerequisites():
            if not self.ask_user_confirmation("Continue anyway?", default=False):
                print("Exiting due to prerequisite issues.")
                return
        
        print("\n🎯 PIPELINE STEPS:")
        print("   1. Generate Cards (square_generator.py)")
        print("   2. Convert to MSE + Images (MTGCG_mse.py)")
        print("      └─ Images handled via config 'image_method' setting")
        
        # Step 1: Card Generation
        print("\n" + "="*50)
        current_cards = self.config["square_config"]["total_cards"]
        current_image_model = self.config["api_params"]["image_model"]
        current_concurrency = self.config["square_config"]["concurrency"]
        
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
                    print(f"Invalid input: {e}")
                    return
            
            if not self.run_square_generator(**overrides):
                if not self.ask_user_confirmation("Continue with remaining steps despite failure?", default=False):
                    print("❌ Stopping pipeline due to card generation failure.")
                    return
        else:
            print("⏭️ Skipping card generation...")
        
        # Step 2: MSE Conversion + Images
        print("\n" + "="*50)
        current_image_method = self.config.get("mtgcg_mse_config", {}).get("image_method", "download")
        if self.ask_user_confirmation(f"📋 Convert cards to MSE format + handle images (method: {current_image_method})?"):
            if not self.run_mse_conversion():
                print("❌ MSE conversion failed.")
        else:
            print("⏭️ Skipping MSE conversion...")
        
        print("\n🎉 PIPELINE COMPLETE!")
        print("="*30)
        
        # Show final results
        self.show_results()
        
    def show_results(self):
        """Display results summary after completion."""
        output_dir = Path(self.config["square_config"]["output_dir"])
        print(f"📁 Check your results in: {output_dir.absolute()}")
        
        # Extract config name for finding the MSE set file
        config_name = os.path.splitext(os.path.basename(self.config_path))[0]
        
        results = []
        if (output_dir / "generated_cards.json").exists():
            results.append("✅ generated_cards.json - Card data")
        
        # Look for MSE set file with config name prefix
        mse_set_file = output_dir / f"{config_name}-mse-out.mse-set"
        if mse_set_file.exists():
            results.append(f"✅ {config_name}-mse-out.mse-set - MSE set file")
        elif (output_dir / "mse-out.mse-set").exists():  # Fallback for old naming
            results.append("✅ mse-out.mse-set - MSE set file")
            
        if (output_dir / "mse-out").exists() and list((output_dir / "mse-out").glob("*.png")):
            png_count = len(list((output_dir / "mse-out").glob("*.png")))
            results.append(f"✅ mse-out/ - {png_count} card images")
        if (output_dir / "forge_out").exists():
            results.append("✅ forge_out/ - Forge format files")
        
        if results:
            print("📊 Generated files:")
            for result in results:
                print(f"   {result}")
        else:
            print("⚠️  No output files detected")
        
        # Show the appropriate MSE file path
        if mse_set_file.exists():
            print(f"\n💡 To view your cards, open {mse_set_file} in Magic Set Editor")
        elif (output_dir / "mse-out.mse-set").exists():
            print(f"\n💡 To view your cards, open {output_dir / 'mse-out.mse-set'} in Magic Set Editor")
    
    def batch_mode(self, steps: list):
        """Run the orchestrator in batch mode with specified steps."""
        print(f"\n🤖 RUNNING BATCH MODE: {' -> '.join(steps)}")
        
        # Run config validation for batch mode (no save)
        self._run_config_validation(save=False)
        
        success = True
        
        if "cards" in steps:
            success &= self.run_square_generator()
        
        if "mse" in steps and success:
            success &= self.run_mse_conversion()
        
        # Note: 'images' step is handled within MTGCG_mse.py based on config
        if "images" in steps:
            print("ℹ️  Images are handled automatically by the MSE conversion step")
            print("   Configure 'mtgcg_mse_config.image_method' in your config file")
        
        if success:
            print("\n🎉 BATCH PROCESSING COMPLETE!")
        else:
            print("\n❌ BATCH PROCESSING FAILED!")
            sys.exit(1)

    def check_mode(self, save: bool = False):
        """Check configuration and display summary without running any steps."""
        print("\n🔍 CONFIGURATION CHECK MODE")
        print("="*50)
        
        # Run full configuration validation with optional save
        self._run_config_validation(save=save)
        
        # Display configuration summary
        self.display_config_summary()
        
        # Check prerequisites 
        print("\n🔧 PREREQUISITE CHECK:")
        prereq_ok = self.check_prerequisites()
        
        if prereq_ok:
            print("✅ All prerequisites satisfied!")
        else:
            print("⚠️  Some prerequisites have issues (see above)")
        
        # Check output directory structure
        print("\n📁 OUTPUT DIRECTORY STRUCTURE:")
        output_dir = Path(self.config["square_config"]["output_dir"])
        config_name = Path(self.config_path).stem
        config_subdir = output_dir / config_name
        
        print(f"   Base output directory: {output_dir}")
        print(f"   Config subdirectory: {config_subdir}")
        print(f"   Cards file would be: {config_subdir / f'{config_name}_cards.json'}")
        print(f"   MSE set file would be: {config_subdir / f'{config_name}-mse-out.mse-set'}")
        
        # Check if output files already exist
        cards_file = config_subdir / f"{config_name}_cards.json"
        mse_file = config_subdir / f"{config_name}-mse-out.mse-set"
        
        print("\n📊 EXISTING OUTPUT FILES:")
        if cards_file.exists():
            print(f"   ✅ Cards file exists: {cards_file}")
        else:
            print(f"   ❌ Cards file not found: {cards_file}")
            
        if mse_file.exists():
            print(f"   ✅ MSE set file exists: {mse_file}")
        else:
            print(f"   ❌ MSE set file not found: {mse_file}")
        
        print(f"\n✅ Configuration check complete for: {self.config_path}")
        print("   Use without --check to run the pipeline.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MerlinAI - MTG Card Generation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --batch cards mse                 # Run all steps
  %(prog)s --batch cards                     # Only generate cards
  %(prog)s my_config.yml --batch mse         # Use custom config, run MSE only
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
        "--batch", 
        nargs="*",
        choices=["cards", "mse", "images"],
        help="Run in batch mode with specified steps (images handled by mse step)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
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
    
    # Validate argument combinations
    if args.save and not args.check:
        parser.error("--save can only be used with --check")
    
    # Setup logging based on verbose flag
    setup_logging(verbose=args.verbose)
    
    # Initialize orchestrator
    if args.list_configs:
        configs_dir = Path(__file__).parent / 'configs'
        if configs_dir.exists():
            files = [p for p in configs_dir.glob('*.yml')
                     if p.name != 'DEFAULTSCONFIG.yml' and not p.name.startswith('__')]
            if not files:
                print("(no user config files found)")
            else:
                for f in sorted(files):
                    print(f.name)
        else:
            print("configs/ directory not found")
        return

    orchestrator = MerlinAIOrchestrator(args.config, verbose=args.verbose)
    
    # Run in appropriate mode
    if args.check:
        orchestrator.check_mode(save=args.save)
    elif args.batch is not None:
        orchestrator.batch_mode(args.batch)
    else:
        orchestrator.interactive_mode()


if __name__ == "__main__":
    main()
