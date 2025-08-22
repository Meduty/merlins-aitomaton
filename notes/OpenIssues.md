# Open Issues - MerlinAI MTG Card Generator

*Last Updated: August 22, 2025*

## Major Consistency Issues

### 1. **Config Schema Inconsistencies** - **HIGH PRIORITY**

**Problem**: Mixed usage of old and new schema throughout codebase
- `square_generator.py` still expects legacy `card_types_weights` structure
- Orchestrator builds this from new schema, but generator redundantly validates
- Creates fragile coupling between validation and generation phases

**Current Code**:
```python
# square_generator.py
if "card_types_weights" not in skeleton_params_full:
    raise ValueError("Missing 'card_types_weights'...")
```

**Fix Needed**: 
- Remove redundant validation in square_generator
- Trust orchestrator validation pipeline
- Ensure forward compatibility for schema evolution

**Files Affected**: `scripts/square_generator.py`, `merlinAI.py`

---

### 2. **Error Handling Inconsistencies** - **HIGH PRIORITY**

**Logging Levels**:
- `scripts/square_generator.py`: Uses `logging.info()` for debug-level information
- `config_manager.py`: Uses structured error/warning/info with emoji prefixes  
- `merlinAI.py`: Basic print statements mixed with logging calls

**Exception Types**:
- Config errors: Mix of `ValueError`, `Exception`, `AssertionError`
- API errors: Inconsistent wrapping/re-raising patterns
- No custom exception hierarchy

**Fix Needed**:
- Create unified exception hierarchy (`MerlinError`, `ConfigError`, `APIError`, `ValidationError`)
- Standardize logging patterns across all modules
- Implement consistent error message formatting

**Files Affected**: All Python files

---

### 3. **Environment Variable Handling** - **HIGH PRIORITY**

**Inconsistent Sources**:
```python
# square_generator.py
API_KEY = os.getenv("API_KEY")           # Direct access
AUTH_TOKEN = os.getenv("AUTH_TOKEN")     

# merlinAI.py  
verbose = os.environ.get("MERLIN_VERBOSE", "1") == "1"  # Different pattern
```

**Issues**:
- No centralized env var management
- Different default value patterns
- Missing validation for required variables
- Inconsistent typing (string vs boolean conversion)

**Fix Needed**:
- Create `EnvConfig` class for centralized environment management
- Validate required variables at startup
- Consistent type conversion patterns

---

### 4. **Path Handling Inconsistencies** - **MEDIUM PRIORITY**

**Problems**:
- `merlinAI.py`: Hardcoded relative paths (`configs_dir = "configs"`)
- `config_manager.py`: Mix of absolute and relative path handling
- No consistent base directory resolution
- Potential issues when running from different working directories

**Fix Needed**:
- Create `ProjectPaths` class for consistent path management
- Use `pathlib.Path` throughout instead of string concatenation
- Establish clear base directory resolution strategy

---

### 5. **Type Annotations Missing/Inconsistent** - **MEDIUM PRIORITY**

**Current State**:
```python
# Some functions fully typed:
def bounded_value_with_rarity(mean: float, low: float, high: float, ...) -> float:

# Others missing types:
def card_skeleton_generator(index, api_params, skeleton_params, config):  # No types

# Inconsistent Optional usage
```

**Fix Needed**:
- Add comprehensive type annotations to all public functions
- Create TypedDict definitions for configuration structures
- Use Literal types for string enums (e.g., `types_mode`)
- Enable mypy checking in CI/development

---

### 6. **Configuration Validation Depth** - **MEDIUM PRIORITY**

**Shallow vs Deep Validation**:
- Config manager: Deep structural validation with drift correction
- Square generator: Basic presence checks only
- No validation of semantic relationships (e.g., `mana_curves` keys match `colors`)
- Missing cross-validation between related config sections

**Fix Needed**:
- Implement semantic validation rules
- Validate relationships between config sections
- Add config schema versioning support
- Create validation test suite

---

### 7. **Progress Reporting Inconsistencies** - **LOW PRIORITY**

**Current State**:
```python
# square_generator.py: Rich progress bars with tqdm
with tqdm(total=total_cards, desc="Generating card information"...

# merlinAI.py: Simple print statements  
print("✓ Configuration validated and normalized successfully")
```

**Fix Needed**:
- Create unified progress/status reporting interface
- Consistent progress bar usage across components
- Structured status messages for better UX

---

## Specific Code Issues

### 1. **SkeletonParams Constructor Brittleness**

**Issue**: Constructor requires ALL parameters with no graceful degradation
```python
def __init__(self, canonical_card_types: Optional[list[str]] = None, ...):
    if canonical_card_types is None:
        raise ValueError("canonical_card_types must be provided")
```

**Problem**: Breaks if config schema evolves; no forward compatibility
**Fix**: Implement builder pattern or config-driven constructor

---

### 2. **Global State Pollution**

**Issue**: Components modify shared state in-place
```python
# square_generator.py modifies api_params.creative in-place
if merlinAI_lib.check_mutation(extra_creative_chance):
    out_params.creative = True  # Mutates shared state
```

**Fix**: Implement immutable config objects or explicit copy semantics

---

### 3. **Inconsistent Normalization Logic**

**Problem**: Two different normalization implementations
```python
# Config manager: Exact 100.0 enforcement
total_weight = sum(row.values())
for k in row:
    row[k] = (row[k] / total_weight) * 100.0

# SkeletonParams: Different normalization logic
def _normalize_row_to_sum(row, total=100.0):
    s = sum(row)
    if s > 0:
        f = total / s
        return [x * f for x in row]
```

**Fix**: Centralize normalization logic in single utility module

---

### 4. **Resource Management**

**Issues**:
- No consistent cleanup of temporary files (ephemeral configs)
- No connection pooling for HTTP requests
- Thread-safe concerns not fully addressed
- Memory usage not monitored for large card generations

**Fix**: Implement context managers and resource cleanup protocols

---

## Architectural Recommendations

### 1. **Centralized Configuration Management**
```python
# config_manager.py enhancement
class ConfigManager:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.getcwd()
        self.env_vars = self._load_env_vars()
    
    def _load_env_vars(self) -> Dict[str, str]:
        """Centralized env var loading with validation"""
        
    def validate_config(self, config: Dict) -> Tuple[Dict, List[str]]:
        """Returns (normalized_config, warnings)"""
```

### 2. **Unified Error Handling**
```python
# errors.py
class MerlinError(Exception):
    """Base exception for all Merlin errors"""
    
class ConfigError(MerlinError):
    """Configuration-related errors"""
    
class APIError(MerlinError):
    """API communication errors"""
    
class ValidationError(MerlinError):
    """Data validation errors"""
```

### 3. **Type Safety**
```python
# types.py
from typing import TypedDict, Literal

class SkeletonConfig(TypedDict):
    types_mode: Literal["normal", "play"]
    power_level: float
    # ... etc

class APIConfig(TypedDict):
    generate_image_prompt: bool
    model: str
    # ... etc
```

### 4. **Consistent Logging**
```python
# logging_config.py
def setup_project_logging(verbose: bool = True, component: str = "merlin"):
    """Unified logging setup for all components"""
    
def get_logger(name: str) -> logging.Logger:
    """Get component-specific logger with consistent formatting"""
```

---

## Testing Consistency Issues

### Missing Test Coverage:
- No unit tests for config validation edge cases
- No integration tests for full pipeline
- No performance benchmarks for normalization logic
- No schema migration tests
- No error handling verification tests

### Needed Test Structure:
```python
# tests/test_config_consistency.py
def test_skeleton_params_from_normalized_config():
    """Ensure SkeletonParams can consume orchestrator output"""
    
def test_config_schema_forward_compatibility():
    """Ensure new schema additions don't break existing logic"""

# tests/test_error_handling.py
def test_unified_exception_hierarchy():
    """Verify all components use consistent error types"""

# tests/test_environment_config.py
def test_env_var_validation():
    """Test environment variable loading and validation"""
```

---

## Priority Implementation Order

### **HIGH PRIORITY** (Immediate Fixes)
1. **Remove redundant validation in `scripts/square_generator.py`**
   - Trust orchestrator validation pipeline
   - Clean up duplicate schema checks
   
2. **Centralize environment variable handling**
   - Create `EnvConfig` class
   - Validate required variables at startup
   
3. **Unify error types and logging patterns**
   - Implement custom exception hierarchy
   - Standardize logging across modules

### **MEDIUM PRIORITY** (Next Sprint)
4. **Add comprehensive type annotations**
   - TypedDict for config structures
   - Enable mypy checking
   
5. **Implement consistent path resolution**
   - Create `ProjectPaths` utility class
   - Use `pathlib.Path` throughout
   
6. **Create unified progress reporting**
   - Consistent status messages
   - Standardized progress bars

### **LOW PRIORITY** (Future Enhancement)
7. **Add performance monitoring/metrics consistency**
8. **Implement comprehensive test coverage**
9. **Create schema migration utilities**
10. **Add resource management improvements**

---

## Dependencies for Fixes

- **Type annotations**: Requires Python 3.9+ for some advanced typing features
- **Path handling**: Already using `pathlib` in some places, extend usage
- **Logging**: May want to consider `structlog` for consistent structured logging
- **Testing**: Will need `pytest` and `pytest-mock` for comprehensive test suite
- **Type checking**: Add `mypy` to development dependencies

---

## Notes

- Many of these issues stem from the rapid evolution of the config schema (old → new format)
- The orchestrator pattern is sound, but enforcement needs to be more consistent
- Consider creating a "compatibility mode" for gradual migration of existing configs
- Some fixes may require breaking changes to internal APIs (acceptable for internal project)
