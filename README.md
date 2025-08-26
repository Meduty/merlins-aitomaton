# Merlin's Aitomaton 🎯

**Merlin's Aitomaton** is a comprehensive AI-powered Magic: The Gathering (MTG) card generation system. It creates custom MTG cards using OpenAI GPT models, exports sets compatible with Magic Set Editor (MSE), and supports image generation via Stable Diffusion. The system features an interactive orchestrator with clean progress visualization, configurable pack builder for realistic booster packs, and robust error handling.

## 📋 Table of Contents

- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [📦 Installation](#-installation)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📝 Usage Guide](#-usage-guide)
- [🔧 Troubleshooting](#-troubleshooting)
- [🏆 Recent Improvements](#-recent-improvements)
- [📄 License](#-license)

---

## 📦 Installation

### Prerequisites

- **Python 3.8+** (tested with Python 3.8-3.11)
- **Git** (for cloning the repository)
- **Text editor** (for configuration files)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Meduty/merlins-aitomaton.git
cd merlins-aitomaton
```

### Step 2: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

**Required packages:**
- `requests` - HTTP requests to MTG Card Generator API
- `tqdm` - Progress bars and visual feedback
- `numpy` - Numerical computations for card parameters
- `pyyaml` - YAML configuration file parsing
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API integration for AI features
- `scipy` - Scientific computing for distribution calculations

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add your API credentials to `.env`:

```env
# Required for card generation via MTG Card Generator API
MTGCG_USERNAME=your_mtgcardgenerator_username
MTGCG_PASSWORD=your_mtgcardgenerator_password

# Required for AI features (OpenAI GPT models)
API_KEY=your_openai_api_key

# Optional - system will auto-login if not provided
AUTH_TOKEN=your_existing_auth_token

# Optional - for MSE image export functionality
MSE_EXE_PATH=path/to/mse.exe
```

**⚠️ Important Notes:**
- Replace the placeholder values with your actual credentials
- **MTGCG credentials**: Sign up at [MTG Card Generator](https://www.mtgcardgenerator.com/)
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- Keep your `.env` file secure and never commit it to version control

### Step 4: Verify Installation

Test your installation with a quick configuration check:

```bash
python merlins_orchestrator.py --check
```

This will:
- ✅ Validate all dependencies are installed
- ✅ Check environment variables are set correctly  
- ✅ Test configuration file loading
- ✅ Verify output directories can be created
- ✅ Show detailed system status

---

## 🚀 Quick Start Guide

### Option 1: Interactive Mode (Recommended for Beginners)

The easiest way to get started:

```bash
python merlins_orchestrator.py
```

This launches an **interactive guided experience** that will:
1. Show your current configuration summary
2. Check all prerequisites automatically
3. Walk you through each step with clear prompts
4. Allow you to modify settings on-the-fly
5. Generate cards with beautiful progress bars

**Example Interactive Session:**
```
🚀 WELCOME TO MERLIN'S AITOMATON - MTG CARD GENERATION ORCHESTRATOR

🔧 CONFIGURATION SUMMARY
📊 Total Cards: 15 (pack_builder enabled)
🔀 Concurrency: 4 threads
📁 Output Directory: output/
🤖 AI Model: gpt-41
🎨 Image Model: none

🔍 CHECKING PREREQUISITES...
✅ All prerequisites satisfied!

🎲 Generate 15 cards using pack builder with 4 threads? [Y/n]: y
📋 Convert to Magic Set Editor format? [Y/n]: y

🎲 RUNNING CARD GENERATION...
Generating card information: 100%|████████████| 15/15 [Elapsed: 02:05 | Avg: 8.3s/card]
✅ Card generation completed successfully!

📋 RUNNING MSE CONVERSION...
[... conversion progress ...]
✅ MSE conversion completed successfully!

🎉 GENERATION COMPLETE! Files saved to: output/test/
```

### Option 2: Direct Execution (Advanced Users)

For direct control and automation:

```bash
# Generate cards only
python merlins_orchestrator.py --module cards

# Full pipeline: cards + MSE conversion
python merlins_orchestrator.py --module cards mse

# With verbose debugging output
python merlins_orchestrator.py --module cards mse --verbose

# Use custom configuration
python merlins_orchestrator.py configs/my_custom_set.yml --module cards mse

# Batch mode - generate multiple iterations
python merlins_orchestrator.py configs/test.yml --batch 5

# Configuration check only (no generation)
python merlins_orchestrator.py configs/my_config.yml --check
```

---

## 📝 Usage Guide

### Basic Workflow

1. **Configure Your Set** (optional - defaults work great!)
2. **Run Generation** (`python merlins_orchestrator.py`)
3. **Review Output** (cards JSON + MSE file in `output/` directory)
4. **Import to MSE** (open the `.mse-set` file in Magic Set Editor)

### Configuration System

Merlin's Aitomaton uses **YAML configuration files** for all settings:

```
configs/
├── DEFAULTSCONFIG.yml    # ← Base defaults (don't modify)
├── test.yml             # ← Simple test configuration
├── maxi01.yml           # ← Goblin vehicle themed set
├── urbanJungle.yml      # ← Modern urban themed set
└── your_custom.yml      # ← Your custom configurations
```

**Key Configuration Sections:**

```yaml
# Card Generation Settings
square_config:
  total_cards: 60          # Number of cards (overridden by pack_builder)
  concurrency: 4           # Parallel generation threads
  output_dir: "output"     # Where files are saved

# AI Model Settings  
api_params:
  model: "gpt-41"          # OpenAI model for card generation
  image_model: "none"      # dall-e-2, dall-e-3, or none
  creative: false          # Extra creative AI responses

# Pack Builder (generates realistic booster packs)
pack_builder:
  enabled: true            # Enable pack-style generation
  pack: [                  # Pack composition
    {"rarity": "common", "count": 7},
    {"rarity": "uncommon", "count": 3},
    {"rarity": {"rare": 6, "mythic": 1}, "count": 1},
    {"type": "basic land", "count": 1}
  ]

# Set Theme and Flavor
set_params:
  set: "My Custom Set"
  themes:
    - "Epic battles"
    - "Ancient powers"
    - "Mystical creatures"
    # ... add your themes
```

### Common Use Cases

#### 1. Generate a Quick Test Set
```bash
python merlins_orchestrator.py configs/test.yml
```
Uses the simple test configuration with pack builder enabled.

#### 2. Create a Themed Set
```yaml
# configs/my_pirate_set.yml
set_params:
  set: "Seas of Adventure"
  themes:
    - "Pirate ships"
    - "Ocean storms"  
    - "Treasure hunting"
    - "Naval combat"
```

```bash
python merlins_orchestrator.py configs/my_pirate_set.yml --module cards mse
```

#### 3. Generate Multiple Variations
```bash
# Generate 5 different versions of the same set
python merlins_orchestrator.py configs/my_set.yml --batch 5
```

Output structure:
```
output/my_set/
├── my_set-1_cards.json + my_set-1-mse-out.mse-set
├── my_set-2_cards.json + my_set-2-mse-out.mse-set  
├── my_set-3_cards.json + my_set-3-mse-out.mse-set
└── ... (etc)
```

#### 4. High-Volume Generation
```yaml
# configs/large_set.yml
square_config:
  total_cards: 200
  concurrency: 8    # Use more threads for faster generation

pack_builder:
  enabled: false    # Disable for exact card count
```

#### 5. Debug Configuration Issues
```bash
# Check configuration without generating anything
python merlins_orchestrator.py configs/my_config.yml --check --verbose

# This shows:
# - Detailed configuration validation
# - Environment variable status  
# - Output directory structure
# - Existing files
# - Prerequisites check results
```

### Output Structure

Each configuration creates organized output directories:

```
output/
├── test/                           # Configuration name becomes directory
│   ├── test_cards.json            # Generated card data (JSON)
│   └── test-mse-out.mse-set       # Magic Set Editor file
├── my_pirate_set/
│   ├── my_pirate_set_cards.json
│   └── my_pirate_set-mse-out.mse-set
└── batch_example/                  # Batch mode creates numbered sets
    ├── batch_example-1_cards.json
    ├── batch_example-1-mse-out.mse-set
    ├── batch_example-2_cards.json
    └── batch_example-2-mse-out.mse-set
```

### Output Modes

**🔇 Clean Mode (Default)** - Perfect for regular use:
```
🎲 RUNNING CARD GENERATION...
Generating card information: 100%|████████████| 15/15 [Elapsed: 02:05 | Avg: 8.3s/card]
✅ Card generation completed successfully!
```

**🔊 Verbose Mode** - Ideal for debugging with `--verbose`:
```
2025-08-25 22:21:31,605 - INFO - ✅ Configuration loaded from configs/test.yml
🎲 RUNNING CARD GENERATION...
2025-08-25 22:21:32,073 - INFO - No auth token found, attempting to login...
2025-08-25 22:21:33,234 - INFO - Authorization token updated successfully.
[... detailed API logs, timing information, etc ...]
Generating card information: 100%|████████████| 15/15 [Elapsed: 02:05 | Avg: 8.3s/card]
✅ Card generation completed successfully!
```

### Advanced Configuration Examples

#### Pack Builder System
Generate realistic MTG booster packs:

```yaml
pack_builder:
  enabled: true
  pack: [
    {"rarity": "common", "count": 10},           # 10 commons
    {"rarity": "uncommon", "count": 3},          # 3 uncommons  
    {"rarity": {"rare": 7, "mythic": 1}, "count": 1}, # 1 rare/mythic (weighted)
    {"type": "basic land", "count": 1},          # 1 basic land
    {"type": "Non-playable", "count": 1}         # 1 token/player aid
  ]
```

#### Image Generation Integration
Configure AI image generation:

```yaml
api_params:
  image_model: "dall-e-3"    # Enable DALL-E image generation

mtgcg_mse_config:
  image_method: "download"   # or "localSD" for Stable Diffusion
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Environment Variable Issues
```bash
# Problem: "Missing environment variable: API_KEY"
# Solution: Check your .env file exists and has correct format
cat .env  # Should show your variables

# Problem: .env file not being read
# Solution: Ensure .env is in the project root directory
ls -la .env  # Should exist in same directory as merlins_orchestrator.py
```

#### Configuration Issues
```bash
# Problem: "Configuration validation failed"
# Solution: Use --check to see detailed validation errors
python merlins_orchestrator.py configs/my_config.yml --check --verbose

# Problem: "DEFAULTSCONFIG.yml not found" 
# Solution: Ensure you're in the project directory and file exists
ls configs/DEFAULTSCONFIG.yml
```

#### API Connection Issues
```bash
# Problem: "MTGCG_USERNAME and MTGCG_PASSWORD must be set"
# Solution: Check credentials are correct and account is active
# Test login manually at https://www.mtgcardgenerator.com/

# Problem: OpenAI API errors
# Solution: Verify API key and check usage limits
# Check https://platform.openai.com/usage
```

#### Permission and Path Issues
```bash
# Problem: "Permission denied" when creating output files
# Solution: Check directory permissions
chmod 755 output/  # Make output directory writable

# Problem: "scripts not found"
# Solution: Ensure you're running from the project root
ls scripts/  # Should show square_generator.py, MTGCG_mse.py, etc.
```

#### Performance and Generation Issues
```bash
# Problem: Generation is very slow
# Solution: Reduce concurrency or total cards
# Edit configs/my_config.yml:
square_config:
  concurrency: 1    # Reduce from 4 to 1
  total_cards: 5    # Start with fewer cards

# Problem: Cards generated but no MSE file
# Solution: Check MSE conversion step completed
python merlins_orchestrator.py --module mse --verbose  # Run just MSE step
```

#### Batch Mode Issues
```bash
# Problem: Batch mode creates subdirectories instead of flat files  
# Current behavior: output/test/test-1/, output/test/test-2/
# This is intended behavior for organization

# Problem: Batch mode fails on iteration 2+
# Solution: Check for config corruption - should be fixed with deep copy
python merlins_orchestrator.py configs/test.yml --batch 2 --verbose
```

### Getting Help

1. **Check Prerequisites**: `python merlins_orchestrator.py --check`
2. **Use Verbose Mode**: Add `--verbose` to see detailed logs
3. **Test Components**: Run each module separately to isolate issues
4. **Check Configuration**: Validate YAML syntax and required fields
5. **Review Output**: Check `output/` directory for partial results

### Debug Commands

```bash
# Full system check with detailed output
python merlins_orchestrator.py --check --verbose

# Test just card generation
python merlins_orchestrator.py --module cards --verbose

# Test configuration loading
python -c "import yaml; print(yaml.safe_load(open('configs/test.yml')))"

# Check environment variables
python -c "import os; print([f'{k}={v[:10]}...' for k,v in os.environ.items() if 'API' in k or 'MTGCG' in k])"
```

## ✨ Features

- **🎛️ Interactive Orchestrator**: Guided pipeline execution with real-time configuration checking, prerequisite validation, and clean progress visualization
- **🤖 AI-Powered Card Generation**: Creates MTG cards using OpenAI GPT models with configurable parameters for colors, rarities, types, and themes
- **📦 Pack Builder System**: Generate realistic booster packs with customizable slot definitions and weighted rarity distribution
- **🎨 Image Generation**: Supports Stable Diffusion (local) and DALL-E with custom prompts and model switching
- **📋 Magic Set Editor Integration**: Converts generated cards into MSE (.mse-set) format for easy import and sharing
- **⚡ Concurrent Processing**: Multi-threaded generation with thread-safe operations and real-time progress tracking
- **🔧 YAML Configuration**: External configuration management with strict validation and CLI overrides
- **� Batch Mode**: Generate multiple iterations with organized output structure
- **📁 Organized Output**: Each configuration creates its own subdirectory preventing overwrites
- **🛡️ Robust Error Handling**: Comprehensive validation with automatic retries and detailed logging

---

## 📁 Project Structure

```
merlins-aitomaton/
├── merlins_orchestrator.py    # 🎯 Main orchestrator (START HERE)
├── configs/
│   ├── DEFAULTSCONFIG.yml     # 📋 Base configuration template
│   ├── test.yml              # 🧪 Simple test configuration
│   ├── maxi01.yml            # 🏆 Goblin vehicle themed set
│   └── *.yml                 # � Your custom configurations
├── scripts/
│   ├── square_generator.py   # 🎲 Core card generation
│   ├── MTGCG_mse.py         # 📋 MSE conversion & export
│   ├── imagesSD.py          # 🎨 Stable Diffusion integration
│   ├── config_manager.py    # ⚙️ Configuration loading & validation
│   └── merlinAI_lib.py      # 🧰 Shared utilities & helpers
├── output/                   # � Generated files (auto-created)
├── requirements.txt          # 📦 Python dependencies
└── README.md                # 📖 This documentation
```
---

## 🏆 Recent Improvements

Merlin's Aitomaton has been significantly modernized with:

### 🔧 **Configuration Externalization**
- Eliminated global variables throughout codebase
- Configuration passed as function parameters 
- Fast-failing validation with descriptive errors
- Strict type checking and required key validation

### 📊 **Clean Progress Visualization**  
- Beautiful progress bars in default mode
- Optional verbose debugging with `--verbose` flag
- Real-time progress tracking for all operations
- Clean, focused output for regular use

### 🎛️ **Comprehensive Orchestrator**
- Interactive and module execution modes
- Prerequisite validation and environment checking
- Runtime configuration display and modification
- Seamless pipeline coordination with error handling

### 🧵 **Threading Safety**
- Thread-safe auth token management with locks
- Concurrent card generation with progress tracking
- Safe configuration sharing across worker threads
- Synchronized metrics collection and reporting

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

- **Author:** Merlin Duty-Knez
- **AI Integration:** OpenAI GPT models, MTG Card Generator API
- **Image Generation:** Stable Diffusion, AUTOMATIC1111
- **Export Format:** Magic Set Editor compatibility
- **Threading & Concurrency:** Python threading with safety locks
- **Configuration System:** YAML-based with strict validation

