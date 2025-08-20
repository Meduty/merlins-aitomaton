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
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    import config_manager  # type: ignore
    from metrics import GenerationMetrics  # type: ignore
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

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
    
    def __init__(self, config_path: str, verbose: bool = False):
        """Initialize orchestrator with configuration."""
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / "scripts"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            config = config_manager.load_config(self.config_path)
            logging.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"❌ Config file not found: {self.config_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"❌ Error loading config: {e}")
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
        """Run the card generation step."""
        print("\n🎲 RUNNING CARD GENERATION...")
        
        # Build command
        cmd = [sys.executable, str(self.scripts_dir / "square_generator.py"), "--config", self.config_path]
        
        # Add CLI overrides
        for key, value in overrides.items():
            if key == "total_cards":
                cmd.extend(["--total-cards", str(value)])
            elif key == "concurrency":
                cmd.extend(["--concurrency", str(value)])
            elif key == "image_model":
                cmd.extend(["--image-model", str(value)])
        
        try:
            if self.verbose:
                logging.info(f"Executing: {' '.join(cmd)}")
            # Use streaming output so progress bars are visible
            result = subprocess.run(cmd, check=True, text=True, env=self._get_subprocess_env())
            
            print("✅ Card generation completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Card generation failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during card generation: {e}")
            return False
    
    def run_mse_conversion(self, method: str = "download") -> bool:
        """Run the Magic Set Editor conversion step."""
        print(f"\n📋 RUNNING MSE CONVERSION (method: {method})...")
        
        cmd = [sys.executable, str(self.scripts_dir / "MTGCG_mse.py"), self.config_path]
        
        try:
            if self.verbose:
                logging.info(f"Executing: {' '.join(cmd)}")
            # Use streaming output so progress bars are visible
            result = subprocess.run(cmd, check=True, text=True, env=self._get_subprocess_env())
            
            print("✅ MSE conversion completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ MSE conversion failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during MSE conversion: {e}")
            return False
    
    def run_image_generation(self) -> bool:
        """Run the Stable Diffusion image generation step."""
        print("\n🎨 RUNNING IMAGE GENERATION...")
        
        # Check if generated_cards.json exists
        output_dir = Path(self.config["square_config"]["output_dir"])
        cards_file = output_dir / "generated_cards.json"
        
        if not cards_file.exists():
            print(f"❌ Card data file not found: {cards_file}")
            print("   Please run card generation first!")
            return False
        
        cmd = [sys.executable, str(self.scripts_dir / "imagesSD.py"), self.config_path]
        
        try:
            if self.verbose:
                logging.info(f"Executing: {' '.join(cmd)}")
            # Use streaming output so progress bars are visible
            result = subprocess.run(cmd, check=True, text=True, env=self._get_subprocess_env())
            
            print("✅ Image generation completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Image generation failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during image generation: {e}")
            return False
    
    def interactive_mode(self):
        """Run the orchestrator in interactive mode."""
        print("\n🚀 WELCOME TO MERLINAI - MTG CARD GENERATION ORCHESTRATOR")
        print("="*65)
        
        # Display configuration summary
        self.display_config_summary()
        
        # Check prerequisites
        if not self.check_prerequisites():
            if not self.ask_user_confirmation("Continue anyway?", default=False):
                print("Exiting due to prerequisite issues.")
                return
        
        print("\n🎯 PIPELINE STEPS:")
        print("   1. Generate Cards (square_generator.py)")
        print("   2. Convert to MSE (MTGCG_mse.py)")
        print("   3. Generate Images (imagesSD.py)")
        
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
        
        # Step 2: MSE Conversion
        print("\n" + "="*50)
        if self.ask_user_confirmation("📋 Convert cards to Magic Set Editor format?"):
            if not self.run_mse_conversion():
                if not self.ask_user_confirmation("Continue with image generation despite MSE failure?", default=True):
                    print("❌ Stopping pipeline due to MSE conversion failure.")
                    return
        else:
            print("⏭️ Skipping MSE conversion...")
        
        # Step 3: Image Generation
        print("\n" + "="*50)
        if self.ask_user_confirmation("🎨 Generate Stable Diffusion images for cards?"):
            if not self.run_image_generation():
                print("❌ Image generation failed.")
        else:
            print("⏭️ Skipping image generation...")
        
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
        
        success = True
        
        if "cards" in steps:
            success &= self.run_square_generator()
        
        if "mse" in steps and success:
            success &= self.run_mse_conversion()
        
        if "images" in steps and success:
            success &= self.run_image_generation()
        
        if success:
            print("\n🎉 BATCH PROCESSING COMPLETE!")
        else:
            print("\n❌ BATCH PROCESSING FAILED!")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MerlinAI - MTG Card Generation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --batch cards mse images          # Run all steps
  %(prog)s --batch cards                     # Only generate cards
  %(prog)s --config my_config.yml --batch mse # Use custom config, run MSE only
        """
    )
    
    parser.add_argument(
        "config", 
        nargs="?", 
        default="configs/config.yml",
        help="Path to configuration file (default: configs/config.yml)"
    )
    
    parser.add_argument(
        "--batch", 
        nargs="*",
        choices=["cards", "mse", "images"],
        help="Run in batch mode with specified steps"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging based on verbose flag
    setup_logging(verbose=args.verbose)
    
    # Initialize orchestrator
    orchestrator = MerlinAIOrchestrator(args.config, verbose=args.verbose)
    
    # Run in appropriate mode
    if args.batch is not None:
        orchestrator.batch_mode(args.batch)
    else:
        orchestrator.interactive_mode()


if __name__ == "__main__":
    main()
