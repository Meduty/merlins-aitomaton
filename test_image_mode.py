#!/usr/bin/env python3
"""
Test script to verify image_mode transformations work correctly.
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import config_manager

def test_image_mode_transformations():
    """Test that image_mode transformations work correctly."""
    
    print("🧪 Testing image_mode transformations...")
    
    # Test cases
    test_cases = [
        ("custom", "dall-e-3", "download"),  # Should remain unchanged
        ("dall-e-2", "dall-e-2", "download"),
        ("dall-e-3", "dall-e-3", "download"), 
        ("localSD", "none", "localSD"),
        ("none", "none", "none")
    ]
    
    for image_mode, expected_image_model, expected_image_method in test_cases:
        # Create minimal test config
        test_config = {
            "aitomaton_config": {
                "image_mode": image_mode
            },
            "api_params": {
                "image_model": "dall-e-3"  # Default from DEFAULTSCONFIG
            },
            "mtgcg_mse_config": {
                "image_method": "download"  # Default from DEFAULTSCONFIG
            }
        }
        
        # Apply transformations
        result = config_manager.apply_image_mode_transformations(test_config.copy())
        
        # Check results
        actual_image_model = result["api_params"]["image_model"]
        actual_image_method = result["mtgcg_mse_config"]["image_method"]
        
        success = (actual_image_model == expected_image_model and 
                  actual_image_method == expected_image_method)
        
        status = "✅" if success else "❌"
        print(f"{status} image_mode='{image_mode}' -> image_model='{actual_image_model}', image_method='{actual_image_method}'")
        
        if not success:
            print(f"   Expected: image_model='{expected_image_model}', image_method='{expected_image_method}'")
    
    print()

def test_with_actual_config():
    """Test with actual DEFAULTSCONFIG.yml loading."""
    
    print("🔧 Testing with actual configuration loading...")
    
    try:
        # Load default config
        default_config = config_manager.load_config(None)
        original_image_model = default_config["api_params"]["image_model"]
        original_image_method = default_config["mtgcg_mse_config"]["image_method"]
        original_image_mode = default_config["aitomaton_config"]["image_mode"]
        
        print(f"📋 DEFAULTSCONFIG.yml settings:")
        print(f"   image_mode: '{original_image_mode}'")
        print(f"   image_model: '{original_image_model}'")
        print(f"   image_method: '{original_image_method}'")
        
        # Test if transformations are applied (should not be for 'custom' mode)
        if original_image_mode == "custom":
            print("✅ image_mode is 'custom' - no transformations should be applied")
        else:
            print(f"🔄 image_mode is '{original_image_mode}' - transformations should be applied")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    
    print()

if __name__ == "__main__":
    test_image_mode_transformations()
    test_with_actual_config()
    print("✨ Testing complete!")
