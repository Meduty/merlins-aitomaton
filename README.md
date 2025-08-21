# MerlinAI 🎯

**MerlinAI** is a comprehensive AI-powered Magic: The Gathering (MTG) card generation and utility suite. It generates custom MTG cards using AI models with externalized configuration, exports sets compatible with Magic Set Editor (MSE), and supports image generation via Stable Diffusion. The system features a complete orchestrator with clean progress bars and verbose debugging modes.

---

## 🚀 Quick Start

### Using the Orchestrator (Recommended)

The easiest way to use MerlinAI is through the main orchestrator:

```bash
# Interactive mode - guided setup with prompts (clean output)
python merlinAI.py

# Verbose mode - full debugging output  
python merlinAI.py --verbose

# Use custom configuration
python merlinAI.py my_config.yml

# Check configuration without running any steps
python merlinAI.py my_config.yml --check

# Batch mode - run all steps automatically with clean progress bars
python merlinAI.py --batch cards mse

# Batch mode with full logging for debugging
python merlinAI.py my_config.yml --batch cards mse --verbose

# Run specific steps only
python merlinAI.py --batch cards         # Only generate cards
python merlinAI.py --batch mse           # Only convert to MSE (includes images)
```

> **📋 Note**: Image generation is handled automatically by the MSE conversion step based on your `mtgcg_mse_config.image_method` setting. Options are:
> - `"download"` - Download images from external API
> - `"localSD"` - Generate with local Stable Diffusion
> - `"none"` - No images (text-only cards)

### Progress Bar Modes

- **Default Mode**: Clean output with only progress bars and essential messages
- **Verbose Mode** (`--verbose`): Full debugging logs plus progress bars

### Manual Execution

You can also run individual components:

```bash
# 1. Generate cards (outputs to output/{config_name}/{config_name}_cards.json)
python scripts/square_generator.py [config.yml]

# 2. Convert to MSE + handle images (based on config image_method)
python scripts/MTGCG_mse.py [config.yml]
```

### Output Organization

Each configuration creates its own organized subdirectory:

```
output/
├── user/                          # Default config outputs
│   ├── user_cards.json           # Generated cards
│   └── user-mse-out.mse-set      # MSE set file
├── my_custom_config/              # Custom config outputs
│   ├── my_custom_config_cards.json
│   └── my_custom_config-mse-out.mse-set
└── another_config/
    ├── another_config_cards.json
    └── another_config-mse-out.mse-set
```

### Configuration Validation

Before running any generation steps, you can validate your configuration:

```bash
# Check configuration without running any steps
python merlinAI.py configs/my_config.yml --check

# This will show:
# - Configuration validation results  
# - Prerequisite checks (API keys, dependencies)
# - Output directory structure
# - Existing output files
# - Detailed config summary
```

The `--check` flag is equivalent to running interactive mode and answering "no" to all generation steps - perfect for debugging configuration issues.

---

## ✨ Features

- **🎛️ Interactive Orchestrator:**  
  Guided pipeline execution with real-time configuration, prerequisite checking, smart error handling, and clean progress visualization.

- **📊 Clean Progress Bars:**  
  Beautiful progress tracking in default mode, with optional verbose logging for debugging.

- **🤖 AI-Powered Card Generation:**  
  Creates MTG cards using OpenAI GPT models via MTG Card Generator API, with configurable parameters for colors, rarities, types, and themes.

- **🎨 Advanced Image Generation:**  
  Supports Stable Diffusion (local), external API downloads, with Lora weights, custom prompts, and model swapping.

- **📋 Magic Set Editor Integration:**  
  Converts generated cards into MSE (.mse-set) format with images and metadata for easy set creation and sharing.

- **🔧 External Configuration Management:**  
  All parameters managed via YAML configuration files with strict validation, fast-failing error handling, and CLI overrides.

- **📁 Organized Output Structure:**  
  Each configuration generates outputs in its own subdirectory, preventing overwrites and keeping projects organized.

- **⚡ Concurrent Processing:**  
  Multi-threaded generation with real-time progress tracking, thread-safe operations, and clean progress visualization.

- **🛡️ Robust Error Handling:**  
  Comprehensive error handling with automatic retries, authentication recovery, and detailed logging.  
  Comprehensive validation, graceful failure recovery, detailed logging (when verbose), and fast-failing configuration validation.

---

## 📁 Project Structure

```
merlinAI/
├── merlinAI.py                 # 🎯 Main orchestrator (START HERE)
├── configs/
│   ├── config.yml             # 🔧 Main configuration file
│   └── DEFAULTSCONFIG.yml     # 📋 Default configuration template
├── scripts/
│   ├── square_generator.py    # 🎲 Core card generation
│   ├── MTGCG_mse.py          # 📋 MSE conversion & export
│   ├── imagesSD.py           # 🎨 Stable Diffusion image generation
│   ├── config_manager.py     # ⚙️ Configuration loading & validation
│   ├── metrics.py            # 📊 Generation metrics & tracking
│   └── merlinAI_lib.py       # 🧰 Shared utilities & helpers
├── output/                    # 📁 Generated files (created automatically)
│   ├── generated_cards.json  # 🃏 Card data
│   ├── mse-out.mse-set       # 📦 Magic Set Editor file
│   ├── mse-out/              # 🖼️ Card images
│   └── forge_out/            # ⚡ Forge format files
├── ORCHESTRATOR_GUIDE.md     # 📖 Detailed orchestrator usage
├── README.md                 # 📝 This file
└── LICENSE                   # 📄 MIT License
```

---

## 🎨 Output Modes

MerlinAI offers two distinct output modes for different use cases:

### 🔇 Clean Mode (Default)
Perfect for regular use - shows only essential information and beautiful progress bars:

```
🤖 RUNNING BATCH MODE: cards

🎲 RUNNING CARD GENERATION...
Generating card information: 100%|████████████| 4/4 [Elapsed: 00:28 | Avg: 7.05s/card]
✅ Card generation completed successfully!
```

### 🔊 Verbose Mode (`--verbose`)
Ideal for debugging - includes all logs, timing, and detailed information:

```
2025-08-20 22:21:31,605 - INFO - ✅ Configuration loaded from configs/config.yml
🤖 RUNNING BATCH MODE: cards

🎲 RUNNING CARD GENERATION...
2025-08-20 22:21:31,605 - INFO - Executing: /usr/bin/python scripts/square_generator.py
2025-08-20 22:21:32,073 - INFO - No auth token found, attempting to login...
2025-08-20 22:21:33,234 - INFO - Authorization token updated successfully.
[... detailed logs ...]
Generating card information: 100%|████████████| 4/4 [Elapsed: 00:28 | Avg: 7.05s/card]
✅ Card generation completed successfully!
```

## 🔧 Configuration System

MerlinAI uses a **strict configuration system** with **fast-failing validation**:

- **📋 YAML Configuration**: All settings in `configs/user.yml` (default) or custom config files
- **🚫 No Global Variables**: Configuration passed as function parameters
- **⚡ Fast Failing**: Missing or invalid config causes immediate errors
- **✅ Type Validation**: All values validated for correct types and ranges
- **🔄 Runtime Loading**: Configuration loaded at execution time from CLI arguments
- **📁 Organized Outputs**: Each config creates its own output subdirectory

### Configuration Files

- **`configs/user.yml`**: Default configuration (rename from old `config.yml`)
- **`configs/DEFAULTSCONFIG.yml`**: Base defaults (do not modify)
- **Custom configs**: Create `configs/myname.yml` for project-specific settings

### CLI Overrides

All scripts support command-line overrides:

```bash
# Override output directory
python scripts/square_generator.py --output-dir /tmp/my_cards

# Override card count and concurrency
python scripts/square_generator.py --total-cards 10 --concurrency 2

# Use custom config with overrides
python scripts/square_generator.py my_project.yml --total-cards 20
```

---

## 🔧 Setup

### 1. Prerequisites

- **Python 3.8+**
- **Required packages:**
  ```bash
  pip install -r requirements.txt
  ```

### 2. Environment Configuration

Create a `.env` file in the project root with your API credentials:

```env
# Required for card generation
MTGCG_USERNAME=your_mtg_username
MTGCG_PASSWORD=your_mtg_password
API_KEY=your_openai_api_key

# Optional - will auto-login if not provided
AUTH_TOKEN=your_auth_token
```

### 5. Configuration

The orchestrator uses `configs/config.yml` by default. You can:

- **Use defaults:** Just run `python merlinAI.py` 
- **Customize settings:** Edit `configs/config.yml`
- **Create custom configs:** Copy and modify for different scenarios

### 3. Stable Diffusion Setup (Optional)

For local image generation, ensure a Stable Diffusion API is running and update the `forge_url_base` in your configuration.

### 4. Required Stable Diffusion Models

MerlinAI comes with a comprehensive configuration for multiple Stable Diffusion models. For the default configuration to work, your Stable Diffusion installation should have these models available:

#### 🎨 **Core Models**
- **`flux1DevHyperNF4Flux1DevBNB_flux1DevHyperNF4`** - Fast FLUX model (Primary)
- **`prefectIllustriousXL_v20p`** - High-quality illustration style
- **`prefectPonyXL_v50`** - Versatile fantasy art
- **`realDream_sdxlPony15`** - Realistic fantasy style
- **`realismByStableYogi_v40FP16`** - Photorealistic generation
- **`sdxlUnstableDiffusers_v9DIVINITYMACHINEVAE`** - Enhanced D&D style
- **`waiNSFWIllustrious_v110`** - Adult content (optional)

#### 🔧 **Required LoRA Models**
- **`FantasyWorldPonyV2`** - Core fantasy enhancement
- **`fantasyV1.1`** - Fantasy world styling
- **`dungeons_and_dragons_xl_v3`** - D&D specific enhancement
- **`Dark_Fantasy_1.5_IL`** - Dark fantasy aesthetics

#### 📝 **Customization Options**

**Option 1: Use Provided Configuration**
Copy the complete configuration from `configs/DEFAULTSCONFIG.yml`:
```bash
cp configs/DEFAULTSCONFIG.yml configs/config.yml
# Edit config.yml to adjust weights, remove unwanted models, etc.
```

**Option 2: Minimal Configuration**
Remove models you don't have and adjust weights:
```yaml
SD_config:
  image_options:
    - name: FLUX_ONLY
      weight: 1.0
      option_params:
        model: FLUX1DEV_HYPERNF4_FLUX1DEVBNB_FLUX1DEVHYPERNF4
        # ... minimal LoRA setup
```

**Option 3: Disable Image Generation**
Set `image_method: "none"` in `mtgcg_mse_config` to skip image generation entirely.

💡 **Note**: The default `DEFAULTSCONFIG.yml` includes all supported models with optimal settings. You can easily remove or modify any models you don't have installed.

---

## 📚 Usage Examples

### Example 1: Quick Start (Clean Mode)
```bash
$ python merlinAI.py --batch cards
🤖 RUNNING BATCH MODE: cards

🎲 RUNNING CARD GENERATION...
Generating card information: 100%|████████████| 4/4 [Elapsed: 00:28 | Avg: 7.05s/card]
✅ Card generation completed successfully!

🎉 BATCH PROCESSING COMPLETE!
```

### Example 2: Full Pipeline with Debugging
```bash
$ python merlinAI.py --batch cards mse images --verbose
2025-08-20 22:21:31,605 - INFO - ✅ Configuration loaded from configs/config.yml

🤖 RUNNING BATCH MODE: cards mse images

🎲 RUNNING CARD GENERATION...
2025-08-20 22:21:31,605 - INFO - Executing: /usr/bin/python scripts/square_generator.py
[... detailed logs ...]
Generating card information: 100%|████████████| 4/4 [Elapsed: 00:28 | Avg: 7.05s/card]
✅ Card generation completed successfully!

📋 RUNNING MSE CONVERSION...
[... MSE conversion logs ...]
✅ MSE conversion completed successfully!

🎨 RUNNING IMAGE GENERATION...
Image 1: 100%|████████████| 100/100 [Elapsed: 00:15 | Speed: 6.67%/s]
[... more image progress bars ...]
✅ Image generation completed successfully!

🎉 BATCH PROCESSING COMPLETE!
```

### Example 3: Interactive Mode
```bash
$ python merlinAI.py
🚀 WELCOME TO MERLINAI - MTG CARD GENERATION ORCHESTRATOR

🔧 CONFIGURATION SUMMARY
📊 Total Cards: 4
🔀 Concurrency: 4
📁 Output Directory: output
🤖 AI Model: gpt-41

🔍 CHECKING PREREQUISITES...
✅ All prerequisites met!

🎲 Generate 4 cards with image model 'dall-e-3' using 4 threads? [Y/n]: y
Modify any settings? [y/N]: n

[... pipeline execution ...]
```

---

## 🎯 Orchestrator Features

### Interactive Mode

The orchestrator provides a guided experience:

```bash
python merlinAI.py
```

**What happens:**
1. **🔧 Configuration Summary:** Shows current settings (cards, models, output dir)
2. **🔍 Prerequisites Check:** Validates environment variables and dependencies  
3. **🎯 Pipeline Steps:** Guides you through each step with clear prompts
4. **⚙️ Runtime Modifications:** Allows you to change settings on-the-fly
5. **📊 Results Summary:** Shows generated files and next steps

### Prerequisites Checking

The system validates:
- ✅ **Required environment variables** (`MTGCG_USERNAME`, `MTGCG_PASSWORD`, `API_KEY`)
- ⚠️ **Optional variables** (`AUTH_TOKEN` - will attempt auto-login if missing)
- 📁 **Output directories** (creates them if missing)
- 📄 **Script files** (ensures all components are present)

### User Information Flow

1. **Configuration Display:**
   ```
   🔧 CONFIGURATION SUMMARY
   📊 Total Cards: 4
   🤖 AI Model: gpt-41
   🎨 Image Model: dall-e-3
   📁 Output Directory: output
   ```

2. **Prerequisites Validation:**
   ```
   🔍 CHECKING PREREQUISITES...
   ⚠️ WARNINGS:
      • Optional environment variable not set: AUTH_TOKEN
   ✅ All prerequisites met!
   ```

3. **Interactive Prompts:**
   ```
   🎲 Generate 4 cards with image model 'dall-e-3'? [Y/n]: 
   Modify any settings? [y/N]:
   Total cards [4]: 8
   Image model (dall-e-3/dall-e-2/none) [dall-e-3]: none
   ```

4. **Progress Feedback:**
   ```
   🎲 RUNNING CARD GENERATION...
   ✅ Card generation completed successfully!
   
   📋 RUNNING MSE CONVERSION...
   ✅ MSE conversion completed successfully!
   ```

5. **Results Summary:**
   ```
   📊 Generated files:
      ✅ generated_cards.json - Card data
      ✅ mse-out.mse-set - MSE set file  
      ✅ mse-out/ - 8 card images
   
   💡 To view your cards, open output/mse-out.mse-set in Magic Set Editor
   ```

### Batch Mode

For automation and scripting:

```bash
# Full pipeline
python merlinAI.py --batch cards mse images

# Selective execution  
python merlinAI.py --batch cards      # Only generate cards
python merlinAI.py --batch mse images # Skip card generation

# Custom configuration
python merlinAI.py my_config.yml --batch cards mse
```

---

## 🛠️ Individual Components

### Card Generation (`square_generator.py`)

Generates MTG cards using AI with threading support:

```bash
python scripts/square_generator.py --config configs/config.yml \
  --total-cards 10 --concurrency 4 --image-model none
```

**Features:**
- 🔀 Multi-threaded generation with thread-safe auth token management
- 🎯 Configurable card parameters (colors, types, rarities, themes)
- 📊 Real-time metrics and progress tracking
- 🛡️ Robust error handling with retry mechanisms
- 🔒 Thread-safe operations for concurrent processing

### MSE Conversion (`MTGCG_mse.py`)

Converts card data to Magic Set Editor format:

```bash
python scripts/MTGCG_mse.py configs/config.yml
```

**Outputs:**
- 📦 `mse-out.mse-set` - Complete MSE set file
- 🖼️ `mse-out/` - Individual card images
- ⚡ `forge_out/` - Forge-compatible format

### Image Generation (`imagesSD.py`)

Generates custom card art using Stable Diffusion:

```bash
python scripts/imagesSD.py configs/config.yml
```

**Features:**
- 🎨 Multiple model support with dynamic switching
- 🔧 Lora weights and custom prompts
- 📈 Progress tracking and error handling
- 🖼️ High-quality image generation

---

## ⚙️ Configuration

### Main Configuration File (`configs/config.yml`)

Key sections:

```yaml
square_config:
  total_cards: 4              # Number of cards to generate
  concurrency: 4              # Parallel threads  
  output_dir: "output"        # Output directory
  sleepy_time: 0.1           # Delay between operations

api_params:
  model: "gpt-41"            # AI model for card generation
  image_model: "dall-e-3"    # Image generation model
  generate_image_prompt: false

skeleton_params:
  colors: ["white", "blue", "black", "red", "green", "colorless"]
  rarities: ["common", "uncommon", "rare", "mythic"]
  # ... detailed card generation parameters

set_params:
  setName: "Custom Set"       # Your set name
  setCode: "CST"             # 3-letter set code
```

### CLI Overrides

Most settings can be overridden via command line:

```bash
python scripts/square_generator.py --config config.yml \
  --total-cards 20 \
  --concurrency 8 \
  --output-dir custom_output \
  --image-model none
```

---

## 🔍 Troubleshooting

### Common Issues

1. **Configuration Errors:**
   - **Fast-failing validation** will immediately show missing or invalid config keys
   - Check `configs/config.yml` against `configs/DEFAULTSCONFIG.yml` for reference
   - All required keys must be present - no fallback defaults
   - Use `--verbose` to see detailed configuration loading

2. **Import Errors in IDE:**
   - The `# type: ignore` comments suppress VS Code linting errors
   - These are false positives due to dynamic path manipulation
   - Code runs correctly despite IDE warnings

3. **Missing Environment Variables:**
   - The orchestrator will clearly indicate missing variables during prerequisite check
   - Required: `MTGCG_USERNAME`, `MTGCG_PASSWORD`, `API_KEY`
   - Optional: `AUTH_TOKEN` (will auto-login if missing)

4. **API Authentication Issues:**
   - Check your MTGCG credentials in `.env`
   - The system will attempt automatic re-authentication
   - Look for "401 Unauthorized" errors in verbose logs

5. **Progress Bar Issues:**
   - Use default mode for clean progress bars only
   - Use `--verbose` if progress bars seem stuck (shows detailed logs)
   - Progress bars work correctly with multi-threading

6. **Threading Issues:**
   - All auth token updates are thread-safe with locks
   - Configuration sharing across threads is safe
   - Metrics collection is synchronized across threads

### Debug Modes

- **Clean Mode (Default)**: `python merlinAI.py` - Only progress bars and essential messages
- **Verbose Mode**: `python merlinAI.py --verbose` - Full debugging output
- **Interactive Mode**: Step-by-step guided execution with confirmations

### Getting Help

- 📖 See `ORCHESTRATOR_GUIDE.md` for detailed usage examples
- 🔧 Check `configs/DEFAULTSCONFIG.yml` for all available options
- 📊 Run with `--verbose` flag for detailed logging and debugging
- 🛡️ Use interactive mode for guided troubleshooting and configuration checking
- ⚡ Configuration errors show immediate, specific error messages

---

## 🏆 Recent Improvements

MerlinAI has been significantly modernized with:

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
- Interactive and batch execution modes
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

