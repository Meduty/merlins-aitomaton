# Configuration Refactoring Guide

This document outlines the comprehensive refactoring of the MTG Card Generator to improve configuration management and eliminate global variables.

## Overview of Changes

The refactoring involved several key improvements:

1. **Configuration Management**: Centralized configuration loading with defaults
2. **Global Variable Elimination**: Replaced global variables with parameter passing
3. **Metrics System**: Created a thread-safe metrics tracking system
4. **Command Line Interface**: Added CLI argument support with overrides

## New File Structure

### New Files Created

- `configs/DEFAULTSCONFIG.yml` - Contains all default configuration values
- `scripts/config_manager.py` - Handles configuration loading and CLI parsing
- `scripts/metrics.py` - Thread-safe metrics tracking without global variables

### Modified Files

- `scripts/square_generator.py` - Refactored to use new config system

## Key Changes

### 1. Configuration System

**Before**: Configuration was loaded globally at module level with hardcoded defaults scattered throughout the code.

**After**: 
- All defaults are centralized in `DEFAULTSCONFIG.yml`
- User config files override defaults via deep merge
- CLI arguments can override both config files
- Configuration is passed as parameters to functions

### 2. Global Variable Elimination

**Before**: Used global variables for metrics tracking and constants:
```python
COLORS = {}
RARITIES = {}
SUCCESSFUL = 0
TOTAL_RUNTIME = 0.0
CANONICAL_CARD_TYPES = merlinAI_lib.CANONICAL_CARD_TYPES
DEFAULT_TYPE_WEIGHTS = merlinAI_lib.DEFAULT_TYPE_WEIGHTS
```

**After**: Created `GenerationMetrics` class with thread-safe access and moved constants to config:
```python
metrics = GenerationMetrics()
metrics.update_color(color_identity)
metrics.increment_successful()

# Constants now loaded from config
skeleton_params = skeletonParams(**config["skeleton_params"])
```

### 3. Function Signatures

**Before**:
```python
def card_skeleton_generator(index, api_params, skeletonParams):
def generate_card(index, api_params):
```

**After**:
```python
def card_skeleton_generator(index, api_params, skeletonParams, config):
def generate_card(index, api_params, metrics, config):
```

### 4. Constants Moved to Configuration

**Before**: Constants imported from `merlinAI_lib`:
```python
CANONICAL_CARD_TYPES = merlinAI_lib.CANONICAL_CARD_TYPES
DEFAULT_TYPE_WEIGHTS = merlinAI_lib.DEFAULT_TYPE_WEIGHTS
```

**After**: All constants defined in `DEFAULTSCONFIG.yml`:
```yaml
skeleton_params:
  canonical_card_types:
    - "creature"
    - "artifact creature"
    # ... etc
  default_type_weights:
    creature: 50
    artifact creature: 0
    # ... etc
```

### 4. Command Line Interface

**New features**:
```bash
python scripts/square_generator.py --config my_config.yml --total-cards 50 --concurrency 8
```

## Usage Examples

### Basic Usage (Defaults Only)
```bash
python scripts/square_generator.py
```
Uses all values from `DEFAULTSCONFIG.yml`

### With Custom Config
```bash
python scripts/square_generator.py --config configs/my_custom.yml
```
Loads defaults, then overrides with custom config

### With CLI Overrides
```bash
python scripts/square_generator.py --config configs/my_custom.yml --total-cards 100 --concurrency 16
```
Loads defaults, applies custom config, then applies CLI overrides

### Create Minimal Custom Config
```yaml
# my_config.yml
square_config:
  total_cards: 50
  concurrency: 8

skeleton_params:
  power_level: 7.0
```
Only specify what you want to change from defaults.

## Benefits of the Refactoring

1. **Maintainability**: All defaults in one place, easier to update
2. **Flexibility**: Easy to create different configurations for different use cases
3. **Thread Safety**: Proper metrics tracking without race conditions
4. **Testability**: Functions can be tested in isolation with mock configs
5. **CLI Convenience**: Quick parameter changes without editing config files
6. **Documentation**: Clear separation of concerns and explicit dependencies
7. **No Hardcoded Constants**: All game data (card types, weights) configurable
8. **Complete Configurability**: Every aspect of card generation can be customized

## Migration from Old Code

If you have existing code that calls the old functions:

**Old way**:
```python
card = generate_card(0, api_params)
```

**New way**:
```python
config = config_manager.load_config("my_config.yml")
metrics = GenerationMetrics()
card = generate_card(0, api_params, metrics, config)
```

## Configuration File Hierarchy

1. **DEFAULTSCONFIG.yml** - Base defaults (never edit this)
2. **User config file** - Your customizations (override defaults)
3. **CLI arguments** - Runtime overrides (highest priority)

This approach ensures you never lose defaults and can easily track what you've customized.

## Testing the Refactoring

The refactoring has been thoroughly tested with:
- ✅ Configuration loading from YAML files
- ✅ Constants moved from code to config
- ✅ Metrics system with thread-safe operations
- ✅ CLI argument parsing and overrides
- ✅ skeletonParams integration with new config system
- ✅ Deep merge of configuration hierarchies
- ✅ Immutable configuration handling (no side effects)
