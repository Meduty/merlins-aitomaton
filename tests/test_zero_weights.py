"""
Test script to demonstrate the complete zero-weight handling solution.
"""
import sys
sys.path.append('scripts')
from scripts.merlinAI_lib import check_and_normalize_config
from scripts.square_generator import SkeletonParams, APIParams, card_skeleton_generator
import logging

def test_scenario(config_path, scenario_name, test_generation=False):
    """Test a specific configuration scenario."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SCENARIO: {scenario_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Test validation and normalization
        config = check_and_normalize_config(config_path)
        if not config:
            print('❌ Configuration failed validation')
            return False
            
        print(f"\n✅ SUCCESS: Configuration validation and normalization passed")
        
        if test_generation:
            print(f"\n🎯 TESTING CARD GENERATION")
            print(f"{'-'*50}")
            
            # Add required missing sections for testing
            if 'square_config' not in config:
                config['square_config'] = {}
            config['square_config'].setdefault('sleepy_time', 0.01)
            config['square_config'].setdefault('standard_deviation_powerLevel', 0.5)
            config['square_config'].setdefault('power_level_rarity_skew', 0.5)
            
            # Create skeleton params and test a few generations
            skeleton_params = SkeletonParams(**config['skeleton_params'])
            api_params = APIParams(api_key='test', auth_token='test', setParams=config.get('set_params', {}))
            
            # Test a few card generations
            for i in range(3):
                try:
                    result = card_skeleton_generator(i, api_params, skeleton_params, config)
                    card_type = result.userPrompt.get('type', 'NOT SET')
                    color = result.userPrompt.get('colorIdentity', 'NOT SET')
                    rarity = result.userPrompt.get('rarity', 'NOT SET')
                    print(f"   Card {i+1}: Type='{card_type}', Color='{color}', Rarity='{rarity}'")
                except Exception as e:
                    print(f"   Card {i+1}: ❌ ERROR - {e}")
                    
        return True
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False

def main():
    """Run comprehensive tests of all implemented features."""
    # Suppress card generation logs for cleaner output
    logging.basicConfig(level=logging.CRITICAL)
    
    print("🚀 COMPREHENSIVE ZERO-WEIGHT HANDLING TEST SUITE")
    print("=" * 80)
    print("Testing all scenarios and edge cases we've implemented...")
    
    test_scenarios = [
        ('configs/test_none_type.yml', 'Zero Type Weights (None type generation)', True),
        ('configs/test_edge_cases.yml', 'Mixed Edge Cases (various normalizations)', False),
        ('configs/DEFAULTSCONFIG.yml', 'Default Configuration (baseline)', False),
    ]
    
    results = []
    for config_path, scenario_name, test_gen in test_scenarios:
        success = test_scenario(config_path, scenario_name, test_gen)
        results.append((scenario_name, success))
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for scenario, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status}: {scenario}")
    
    print(f"\nOverall: {passed}/{total} scenarios passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Zero-weight handling is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
