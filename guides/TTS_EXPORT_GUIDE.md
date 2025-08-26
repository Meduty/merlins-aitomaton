# TTS (Tabletop Simulator) Export

The TTS export functionality has been enhanced to provide a complete pipeline for creating Tabletop Simulator deck files using the `tts-deckconverter` utility.

## Overview

The export process now supports two modes:

1. **Images Only** (`--mode images`): Export card images from MSE files (original functionality)
2. **Complete TTS Export** (`--mode complete`): Full pipeline including image export, deck list creation, and TTS conversion (new)

## Prerequisites

### Required Software

1. **Magic Set Editor (MSE)**: For exporting card images
   - Set `MSE_PATH` environment variable to point to `mse.exe`

2. **tts-deckconverter**: For converting deck lists to TTS format
   - Download from: https://github.com/jeandeaual/tts-deckconverter
   - Add to your system PATH
   - Or place the executable in a directory that's in your PATH

### Installation of tts-deckconverter

**Option 1: Download Binary**
1. Go to https://github.com/jeandeaual/tts-deckconverter/releases
2. Download the appropriate binary for your system
3. Add it to your PATH

**Option 2: Build from Source**
```bash
git clone https://github.com/jeandeaual/tts-deckconverter.git
cd tts-deckconverter
go build
```

## Usage

### Command Line

```bash
# Complete TTS export (default)
python scripts/exportToTTS.py configs/test.yml

# Complete TTS export with custom settings
python scripts/exportToTTS.py configs/test.yml --mode complete --deck-name "My Custom Deck" --output-dir "./custom_output"

# Images only
python scripts/exportToTTS.py configs/test.yml --mode images

# With verbose logging
python scripts/exportToTTS.py configs/test.yml --verbose
```

### From Orchestrator

The orchestrator automatically uses the complete TTS export when running the full pipeline or in batch mode.

## How It Works

### Complete TTS Export Pipeline

1. **Image Export**: Uses MSE to export card images as PNG files
2. **Deck List Creation**: Creates a text file in tts-deckconverter format:
   ```
   /path/to/image1.png (Card Name 1)
   /path/to/image2.png (Card Name 2)
   /path/to/image3.png (Card Name 3)
   ```
3. **TTS Conversion**: Uses tts-deckconverter to create the final TTS deck JSON

### Output Structure

```
output/
└── config_name/
    └── tts_export/
        ├── images/
        │   ├── Card1.png
        │   ├── Card2.png
        │   └── ...
        ├── config_name_deck.txt
        └── [TTS JSON files from tts-deckconverter]
```

## Deck List Format

The deck list file uses the tts-deckconverter custom card format:

```
<Count> <Image Path> (<Card Name>)
```

- **Count**: Optional, defaults to 1 (currently always 1 for generated cards)
- **Image Path**: Absolute path to the card image
- **Card Name**: Name of the card in parentheses

Example:
```
C:\Users\User\Output\test\tts_export\images\Lightning Bolt.png (Lightning Bolt)
C:\Users\User\Output\test\tts_export\images\Counterspell.png (Counterspell)
C:\Users\User\Output\test\tts_export\images\Giant Growth.png (Giant Growth)
```

## Configuration

The TTS export uses the same configuration as the rest of the pipeline. No additional configuration is required.

## Troubleshooting

### Common Issues

1. **"tts-deckconverter not found"**
   - Make sure tts-deckconverter is installed and in your PATH
   - Test with: `tts-deckconverter --version`

2. **"MSE executable not found"**
   - Set the `MSE_PATH` environment variable
   - Add to your `.env` file: `MSE_PATH=C:\Path\To\mse.exe`

3. **"Cards JSON file not found"**
   - Make sure you've run the card generation step first
   - The cards JSON file should be in `output/config_name/config_name_cards.json`

4. **"Image not found for card"**
   - Make sure the MSE export step completed successfully
   - Check that image files exist in the images directory
   - Card names in JSON should match image filenames

### Verbose Output

Use `--verbose` flag to see detailed logging of each step:

```bash
python scripts/exportToTTS.py configs/test.yml --verbose
```

This will show:
- MSE command execution details
- Deck list creation progress
- tts-deckconverter command and output
- File paths and counts

## API Reference

### Functions

- `export_complete_tts_deck()`: Complete TTS export pipeline
- `export_card_images_with_mse()`: MSE image export only
- `create_tts_deck_list()`: Create deck list file
- `convert_to_tts_deck()`: Run tts-deckconverter

### Parameters

- `config`: Configuration dictionary
- `config_path`: Path to configuration file
- `output_dir`: Custom output directory (optional)
- `deck_name`: Custom deck name for TTS (optional)
- `mode`: Export mode ("complete" or "images")
