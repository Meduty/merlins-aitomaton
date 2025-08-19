# merlinAI

**merlinAI** is an AI-powered Magic: The Gathering (MTG) card generator and utility suite. It generates custom MTG cards using AI models, manages configuration, and exports sets compatible with Magic Set Editor (MSE). It also supports image generation via Stable Diffusion or external APIs.

---

## Features

- **Automated Card Generation:**  
  Generates MTG cards using AI (OpenAI GPT models) and the MTG Card Generator API, with configurable parameters for color, rarity, type, and more.

- **Image Generation:**  
  Supports generating card art using Stable Diffusion (local) or downloading from external sources, with advanced prompt and Lora support.

- **MSE Export:**  
  Converts generated cards into Magic Set Editor (.mse-set) format, including images and metadata, for easy set creation and sharing.

- **Configurable Workflows:**  
  All parameters (card skeleton, image options, API settings) are managed via a single `config.yml` file.

- **Concurrency & Progress Tracking:**  
  Multi-threaded generation and downloading with progress bars and logging.

---

## Project Structure

```
.
├── .env                  # Environment variables (API keys, credentials)
├── config.yml            # Main configuration file
├── square_generator.py   # Main card generation script
├── imagesSD.py           # Stable Diffusion image generation utilities
├── MTGCG_mse.py          # MSE export and image packaging
├── merlinAI_lib.py       # Shared library (weights, helpers, normalization)
├── README.md             # This file
├── LICENSE               # MIT License
└── generated_cards.json  # (Generated) Output cards
```

---

## Setup

1. **Install Dependencies**

   - Python 3.8+
   - Required packages:  
     ```
     pip install requests tqdm numpy pyyaml python-dotenv openai scipy
     ```

2. **Configure Environment**

   - Copy `.env` and fill in your API keys and credentials:
     ```
     MTGCG_USERNAME = "your_username"
     MTGCG_PASSWORD = "your_password"
     API_KEY = "your_openai_api_key"
     AUTH_TOKEN = None
     ```

   - Edit `config.yml` to adjust card generation, image, and export settings.

3. **(Optional) Stable Diffusion**

   - For local image generation, ensure a Stable Diffusion API is running and update `forge_url_base` in `config.yml`.

---

## Usage

### 1. Generate Cards

Run the card generator to create a batch of cards:

```
python square_generator.py
```

- Outputs cards to `generated_cards.json`.

### 2. Generate Images

Generate images for the cards using Stable Diffusion or download:

```
python imagesSD.py
```

- Images are saved to the configured output directory.

### 3. Export to Magic Set Editor

Convert cards and images to MSE format:

```
python MTGCG_mse.py
```

- Produces an `.mse-set` archive in `out/mse-out.mse-set`.

---

## Configuration

All major options are in [config.yml](config.yml):

- **square_config:** Number of cards, concurrency, power level, etc.
- **SD_config:** Image generation models, Lora weights, prompt options.
- **api_params:** AI model selection, prompt/explanation toggles.
- **skeleton_params:** Card color/type/rarity weights, function tags, themes.

See comments in [config.yml](config.yml) for details.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Credits

- Author: Merlin Duty-Knez
- Uses OpenAI, Stable Diffusion, and Magic

