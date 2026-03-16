# Lens Font Recognition Model

<br>

<p align="center">
    <img src="https://static.mixfont.com/assets/20260316-200114-mixfont-lens-banner-qlm9gt3n.webp" alt="Lens Font Recognition Model Banner" />
<p>

<p align="center">
🖥️ <a href="https://www.mixfont.com/models/lens">Lens Demo</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://www.mixfont.com/docs">Lens Commercial API</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://github.com/mixfont/lens">Github</a>

</p>

Lens is a neural-net based font recognition and classification model. It has been specifically trained on open-source fonts. It recognizes the closest matching open-source font for the largest word in a provided image. Lens supports most [Google Fonts](https://fonts.google.com/) families, but also includes other open source fonts outside of the Google Fonts collection. In total is recognizes over 1000 different font families, and over 5000 font variants, including multiple weights and italic style combinations. This repo contains a standalone release of the Lens model and its local inference code. Enjoy!

## News

- 2026.3.16: 🎉🎉🎉 We have released [Lens](https://wwww.mixfont.com/models/lens) as an open-weight font recognition model!

## Contents <!-- omit in toc -->

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Released Models Description and Download](#released-models-description-and-download)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [Python Package Usage](#python-package-usage)
    - [Custom Voice Generation](#custom-voice-generate)
    - [Voice Design](#voice-design)
    - [Voice Clone](#voice-clone)
    - [Voice Design then Clone](#voice-design-then-clone)
    - [Tokenizer Encode and Decode](#tokenizer-encode-and-decode)
  - [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [DashScope API Usage](#dashscope-api-usage)
- [vLLM Usage](#vllm-usage)
- [Fine Tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview

### Introduction

Lens is a powerful font recognition model built on top of Resnet18 to identify and classify fonts from images. It is designed to provide accurate font recognition results, making it a valuable tool for designers, typographers, and anyone interested in identifying fonts or the closest matching font in a given image.

- **Trained only on open-source fonts**: Unlike other font recognition models that do a database lookup on proprietary fonts, Lens is trained on a large dataset of open-source fonts, including most Google Fonts families and other open source fonts. This allows users to freely use the recognized fonts in their projects without worrying about licensing issues.
- **Handles multiple font weights and styles**: Lens is trained on a combination of font weights and styles which makes it more robust and accurate in recognizing fonts in various formats. Lens can still identify a specific font family even if the style is in italic or the font weight is extra bold.
- **Robustness with obscured text, background images, and more**: Lens is built to handle a wide variety of real-world scenarios, including images with obscured text, complex backgrounds, and varying distortion levels. This makes it a versatile tool for font recognition in diverse contexts.
- **Handles images with multiple fonts**: The inference pipeline in Lens first identifies the largest word in the image using OCR, and then runs font recognition on that word. This allows it to handle images with multiple fonts, and still return accurate predictions for the most prominent font in the image.

# Lens

This directory is a standalone open-source release of the Lens font recognition
model and its local inference code.

It contains only the pieces required to:

- download an image from a URL
- run OCR to find the largest word
- classify the word image with the bundled model
- return the same JSON response shape as `lens_modal_app.py`

It does not include Modal deployment code, API server code, or web request
handlers beyond downloading the input image URL for inference.

## Files

- `run_inference.py`: CLI entrypoint
- `lens_inference.py`: OCR + preprocessing + model inference pipeline
- `ocr_word_detection.py`: OCR box detection helpers
- `font_metadata_mapper.py`: maps predicted labels to font metadata entries
- `model/`: bundled trained model artifacts from `2026-03-04_23-50-51`

## Setup

Install Tesseract OCR first.

macOS:

```bash
brew install tesseract
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Then create a virtual environment and install Python dependencies with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
python run_inference.py "https://example.com/image.png"
```

Optional arguments:

- `--top-k 5`
- `--debug`

Pass `--debug` to clear any existing files in `debug/` and write:

- `01_downloaded_image.png`
- `02_word_detection_box.png`
- `03_model_input_crop.png`

## Output

The script prints JSON to stdout with the same top-level keys returned by the
deployed app:

```json
{
  "word": "Example",
  "word_box": {
    "left": 100,
    "top": 120,
    "width": 210,
    "height": 52
  },
  "input_image": {
    "width": 1280,
    "height": 720,
    "image_url": "https://example.com/image.png"
  },
  "font_matches": [
    {
      "name": "Inter",
      "score": 0.83,
      "fonts": [
        {
          "full_name": "Inter Regular",
          "style": "normal",
          "weight": 400,
          "url": "https://static.mixfont.com/fonts/inter/Inter-Regular.ttf"
        }
      ]
    }
  ]
}
```

If OCR cannot find a usable word, the script falls back to running inference on
the original image and still returns predictions.
