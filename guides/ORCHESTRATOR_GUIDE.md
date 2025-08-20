# 🎯 MerlinAI Orchestrator Usage Guide

The main `merlinAI.py` script serves as a comprehensive orchestrator for the entire MTG card generation pipeline.

## 🚀 Quick Start

### Interactive Mode (Recommended)
```bash
python merlinAI.py
```
This launches an interactive interface that guides you through each step with helpful prompts and configuration summaries.

### Batch Mode
```bash
# Run all steps
python merlinAI.py --batch cards mse images

# Run only card generation
python merlinAI.py --batch cards

# Run MSE conversion and image generation
python merlinAI.py --batch mse images
```

### Custom Configuration
```bash
# Use a custom config file
python merlinAI.py my_custom_config.yml

# Batch mode with custom config
python merlinAI.py my_custom_config.yml --batch cards mse
```

## 📊 Pipeline Steps

1. **🎲 Card Generation** (`square_generator.py`)
   - Generates MTG card data using AI
   - Configurable card count, concurrency, and image models
   - Outputs: `generated_cards.json`

2. **📋 MSE Conversion** (`MTGCG_mse.py`)
   - Converts card data to Magic Set Editor format
   - Downloads card images
   - Outputs: `mse-out.mse-set`, `mse-out/` folder with images

3. **🎨 Image Generation** (`imagesSD.py`)
   - Generates custom card images using Stable Diffusion
   - Requires existing card data from step 1
   - Enhanced image quality and customization

## 🔧 Features

- **Prerequisites Check**: Validates environment variables and dependencies
- **Configuration Summary**: Shows current settings before execution
- **Interactive Prompts**: Asks for confirmation and allows real-time modifications
- **Error Handling**: Graceful failure handling with continuation options
- **Progress Tracking**: Real-time feedback on pipeline execution
- **Results Summary**: Shows generated files and helpful next steps

## 📋 Prerequisites

### Required Environment Variables
```bash
export MTGCG_USERNAME="your_username"
export MTGCG_PASSWORD="your_password" 
export API_KEY="your_openai_api_key"
```

### Optional Environment Variables
```bash
export AUTH_TOKEN="your_auth_token"  # Will auto-login if not set
```

## 💡 Tips

- Use interactive mode for first-time setup and testing
- Use batch mode for automated/scripted workflows
- The orchestrator automatically creates output directories
- Configuration can be modified on-the-fly in interactive mode
- Each step can be run independently if needed

## 🎯 Example Workflow

```bash
# 1. Interactive setup and testing
python merlinAI.py

# 2. Once satisfied, use batch mode for production
python merlinAI.py --batch cards mse images

# 3. View results in Magic Set Editor
# Open output/mse-out.mse-set in MSE
```
