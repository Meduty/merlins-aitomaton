#!/usr/bin/env python3
"""
MTG Card Generator - Main Entry Point

This is the main entry point for the MTG Card Generator application.
It simply calls the main function from the square_generator module.
"""

import sys
from pathlib import Path

# Add the scripts directory to the Python path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import and run the main application
if __name__ == "__main__":
    from square_generator import *
    # The main() code is already in the if __name__ == "__main__" block of square_generator.py
    # So we just need to import it and it will run automatically
