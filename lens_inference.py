"""
Standalone OCR + font inference logic for the open-source Lens release.

This module performs:
1. Download image bytes from an HTTP(S) URL.
2. OCR to find the largest confident word bounding box.
3. Resize/pad/normalize with the same preprocessing used during inference.
4. Model inference that returns the same response shape as the deployed app.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from font_metadata_mapper import load_font_metadata_lookup
from ocr_word_detection import (
  WordBox,
  extract_largest_word,
  map_box_to_original_image,
)

if TYPE_CHECKING:
  from PIL import Image


DEFAULT_IMAGE_SIZE = 224
DEFAULT_TOP_K = 3
DEFAULT_PAD_RATIO = 0.25
DEFAULT_MIN_WORD_LEN = 3
DEFAULT_MIN_CONF = -1.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

MODEL_DIR_ENV = "LENS_MODEL_DIR"
MODEL_PATH_ENV = "LENS_MODEL_PATH"


@dataclass(frozen=True)
class ModelBundle:
  model: Any
  idx_to_class: dict[int, str]
  font_metadata_lookup: dict[str, list[dict[str, object]]]
  preprocess: Any
  tensor_transform: Any
  device: Any
  image_height: int
  image_width: int


_MODEL_CACHE: dict[str, ModelBundle] = {}
_DEFAULT_FONT_STYLE = "normal"
_DEFAULT_FONT_WEIGHT = 400
_DEFAULT_FONT_URL = ""
_DEBUG_ORIGINAL_IMAGE_NAME = "01_downloaded_image.png"
_DEBUG_WORD_BOX_IMAGE_NAME = "02_word_detection_box.png"
_DEBUG_MODEL_INPUT_IMAGE_NAME = "03_model_input_crop.png"


class ResizePadToSize:
  """Resize to fit inside a target rectangle and pad without cropping."""

  def __init__(
    self,
    target_height: int,
    target_width: int,
    fill: int | tuple[int, int, int] | str = "sample",
  ) -> None:
    self.target_height = target_height
    self.target_width = target_width
    self.fill = fill

  def _sample_border_color(self, image: "Image.Image") -> tuple[int, int, int]:
    width, height = image.size
    if width == 0 or height == 0:
      return (255, 255, 255)
    pixels = image.load()
    step_x = max(1, width // 50)
    step_y = max(1, height // 50)
    total_r = total_g = total_b = 0
    count = 0
    for x in range(0, width, step_x):
      r, g, b = pixels[x, 0]
      total_r += r
      total_g += g
      total_b += b
      count += 1
      if height > 1:
        r, g, b = pixels[x, height - 1]
        total_r += r
        total_g += g
        total_b += b
        count += 1
    for y in range(0, height, step_y):
      r, g, b = pixels[0, y]
      total_r += r
      total_g += g
      total_b += b
      count += 1
      if width > 1:
        r, g, b = pixels[width - 1, y]
        total_r += r
        total_g += g
        total_b += b
        count += 1
    if count == 0:
      return (255, 255, 255)
    return (
      int(round(total_r / count)),
      int(round(total_g / count)),
      int(round(total_b / count)),
    )

  def __call__(self, image: "Image.Image") -> "Image.Image":
    from torchvision import transforms
    from torchvision.transforms import functional as F

    image = image.convert("RGB")
    width, height = image.size
    if width == 0 or height == 0:
      return image

    fill = self.fill
    if fill == "sample":
      fill = self._sample_border_color(image)

    scale = min(self.target_width / width, self.target_height / height)
    new_width = min(self.target_width, max(1, int(round(width * scale))))
    new_height = min(self.target_height, max(1, int(round(height * scale))))
    image = F.resize(
      image,
      (new_height, new_width),
      interpolation=transforms.InterpolationMode.BICUBIC,
    )
    pad_left = (self.target_width - new_width) // 2
    pad_top = (self.target_height - new_height) // 2
    pad_right = self.target_width - new_width - pad_left
    pad_bottom = self.target_height - new_height - pad_top
    return F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def download_image_bytes(
  image_url: str,
  timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
  max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> bytes:
  parsed = urlsplit(image_url)
  if parsed.scheme not in {"http", "https"}:
    raise ValueError("image_url must use http:// or https://")

  request = Request(
    image_url,
    headers={
      "User-Agent": "lens/1.0",
      "Accept": "image/*",
    },
  )
  try:
    with urlopen(request, timeout=timeout_seconds) as response:
      status = getattr(response, "status", 200)
      if status >= 400:
        raise ValueError(f"image_url request failed with status {status}")
      content_type = (response.headers.get("Content-Type") or "").lower()
      if content_type and not content_type.startswith("image/"):
        raise ValueError(f"image_url did not return an image (got {content_type})")
      content_length = response.headers.get("Content-Length")
      if content_length:
        try:
          content_length_value = int(content_length)
        except ValueError:
          content_length_value = None
        if (
          content_length_value is not None
          and content_length_value > max_download_bytes
        ):
          raise ValueError(
            f"image_url exceeds max size of {max_download_bytes} bytes"
          )
      image_bytes = response.read(max_download_bytes + 1)
  except HTTPError as exc:
    raise ValueError(f"image_url request failed with HTTP {exc.code}") from exc
  except URLError as exc:
    raise ValueError(f"image_url request failed: {exc.reason}") from exc

  if not image_bytes:
    raise ValueError("image_url returned an empty response")
  if len(image_bytes) > max_download_bytes:
    raise ValueError(f"image_url exceeds max size of {max_download_bytes} bytes")
  return image_bytes


def load_image_from_bytes(image_bytes: bytes) -> "Image.Image":
  try:
    from PIL import Image
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pillow to run image inference."
    ) from exc

  def to_rgb_with_white_background(image: "Image.Image") -> "Image.Image":
    has_alpha = image.mode in {"RGBA", "LA"} or (
      image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
      return image.convert("RGB")

    # Composite transparency on white so transparent PNG regions stay readable.
    rgba_image = image.convert("RGBA")
    white_background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_background, rgba_image)
    return composited.convert("RGB")

  try:
    with Image.open(io.BytesIO(image_bytes)) as image:
      return to_rgb_with_white_background(image)
  except Exception as exc:
    raise ValueError("Downloaded data is not a valid image file.") from exc


def _draw_word_box_overlay(
  image: "Image.Image",
  word_box: WordBox | None,
) -> "Image.Image":
  try:
    from PIL import ImageDraw
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pillow to run image inference."
    ) from exc

  overlay = image.convert("RGB")
  if word_box is None:
    return overlay

  draw = ImageDraw.Draw(overlay)
  line_width = max(2, int(round(min(overlay.size) * 0.004)))
  draw.rectangle(
    (word_box.left, word_box.top, word_box.right, word_box.bottom),
    outline=(64, 255, 96),
    width=line_width,
  )
  return overlay


def write_debug_images(
  debug_dir: str | Path | None,
  original_image: "Image.Image",
  word_box: WordBox | None,
  model_input_image: "Image.Image",
) -> None:
  resolved_debug_dir = _as_path(debug_dir)
  if resolved_debug_dir is None:
    return

  if resolved_debug_dir.exists():
    for entry in resolved_debug_dir.iterdir():
      if entry.is_dir():
        shutil.rmtree(entry)
      else:
        entry.unlink()
  resolved_debug_dir.mkdir(parents=True, exist_ok=True)
  original_image.save(resolved_debug_dir / _DEBUG_ORIGINAL_IMAGE_NAME)
  _draw_word_box_overlay(
    image=original_image,
    word_box=word_box,
  ).save(resolved_debug_dir / _DEBUG_WORD_BOX_IMAGE_NAME)
  model_input_image.save(resolved_debug_dir / _DEBUG_MODEL_INPUT_IMAGE_NAME)


def pick_device():
  try:
    import torch
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install torch and torchvision to run inference."
    ) from exc

  if torch.backends.mps.is_available():
    return torch.device("mps")
  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def build_model(num_classes: int):
  from torch import nn
  from torchvision import models

  # Avoid downloading pretrained weights; we load trained checkpoint weights.
  model = models.resnet18(weights=None)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model


def load_class_mapping(model_path: Path, checkpoint: dict) -> dict[str, int]:
  class_to_idx = checkpoint.get("class_to_idx")
  if isinstance(class_to_idx, dict) and class_to_idx:
    return class_to_idx

  classes_path = model_path.parent / "classes.json"
  if classes_path.exists():
    with classes_path.open("r", encoding="utf-8") as file:
      data = json.load(file)
    class_to_idx = data.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
      return class_to_idx

  raise ValueError("Class mapping not found in checkpoint or classes.json")


def resolve_image_dimensions(
  checkpoint: dict,
  fallback: int,
) -> tuple[int, int]:
  image_height = checkpoint.get("image_height")
  image_width = checkpoint.get("image_width")
  if (
    isinstance(image_height, int)
    and image_height > 0
    and isinstance(image_width, int)
    and image_width > 0
  ):
    return image_height, image_width
  image_size = checkpoint.get("image_size")
  if isinstance(image_size, int) and image_size > 0:
    return image_size, image_size
  return fallback, fallback


def build_transforms(image_height: int, image_width: int):
  from torchvision import transforms
  from torchvision.models import ResNet18_Weights

  weights = ResNet18_Weights.DEFAULT
  default_mean = (0.485, 0.456, 0.406)
  default_std = (0.229, 0.224, 0.225)
  meta = getattr(weights, "meta", {}) or {}
  mean = meta.get("mean", default_mean)
  std = meta.get("std", default_std)

  preprocess = transforms.Compose(
    [
      ResizePadToSize(image_height, image_width),
      transforms.Grayscale(num_output_channels=3),
    ]
  )
  tensor_transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
    ]
  )
  return preprocess, tensor_transform


def _as_path(path_value: str | Path | None) -> Path | None:
  if path_value is None:
    return None
  return path_value if isinstance(path_value, Path) else Path(path_value)


def resolve_model_path(
  model_root: Path,
  model_dir: str | Path | None = None,
  model_path: str | Path | None = None,
) -> Path:
  resolved_model_dir = _as_path(model_dir)
  resolved_model_path = _as_path(model_path)
  if resolved_model_dir and resolved_model_path:
    raise ValueError("Provide only one of model_dir or model_path.")

  env_model_path = _as_path(os.getenv(MODEL_PATH_ENV))
  env_model_dir = _as_path(os.getenv(MODEL_DIR_ENV))
  if resolved_model_path is None and env_model_path is not None:
    resolved_model_path = env_model_path
  if resolved_model_dir is None and env_model_dir is not None:
    resolved_model_dir = env_model_dir

  if resolved_model_path is not None:
    return resolved_model_path
  if resolved_model_dir is not None:
    return resolved_model_dir / "font_classifier.pt"

  if model_root.is_file():
    return model_root
  if not model_root.exists():
    raise FileNotFoundError(f"Model root not found: {model_root}")
  if (model_root / "font_classifier.pt").exists():
    return model_root / "font_classifier.pt"

  model_dirs = sorted(
    [
      directory
      for directory in model_root.iterdir()
      if directory.is_dir() and (directory / "font_classifier.pt").exists()
    ]
  )
  if not model_dirs:
    raise FileNotFoundError(f"No model checkpoint found under: {model_root}")
  return model_dirs[-1] / "font_classifier.pt"


def load_model_bundle(model_path: Path) -> ModelBundle:
  try:
    import torch
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install torch and torchvision to run inference."
    ) from exc

  if not model_path.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

  checkpoint = torch.load(model_path, map_location="cpu")
  if not isinstance(checkpoint, dict):
    raise ValueError("Unexpected checkpoint format")

  model_state = checkpoint.get("model_state")
  if not isinstance(model_state, dict) or not model_state:
    raise ValueError("Model weights not found in checkpoint")

  class_to_idx = load_class_mapping(model_path, checkpoint)
  idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
  image_height, image_width = resolve_image_dimensions(checkpoint, DEFAULT_IMAGE_SIZE)

  device = pick_device()
  model = build_model(len(class_to_idx)).to(device)
  model.load_state_dict(model_state)
  model.eval()

  preprocess, tensor_transform = build_transforms(image_height, image_width)
  font_metadata_lookup = load_font_metadata_lookup(model_path)
  return ModelBundle(
    model=model,
    idx_to_class=idx_to_class,
    font_metadata_lookup=font_metadata_lookup,
    preprocess=preprocess,
    tensor_transform=tensor_transform,
    device=device,
    image_height=image_height,
    image_width=image_width,
  )


def get_model_bundle(model_path: Path) -> ModelBundle:
  cache_key = str(model_path)
  if cache_key in _MODEL_CACHE:
    return _MODEL_CACHE[cache_key]
  bundle = load_model_bundle(model_path)
  _MODEL_CACHE[cache_key] = bundle
  return bundle


def run_model(
  image: "Image.Image",
  bundle: ModelBundle,
  top_k: int,
) -> list[dict[str, Any]]:
  import torch

  def coerce_font_entries(
    font_name: str,
    value: object,
  ) -> list[dict[str, object]]:
    if not isinstance(value, list):
      return [
        {
          "full_name": font_name,
          "style": _DEFAULT_FONT_STYLE,
          "weight": _DEFAULT_FONT_WEIGHT,
          "url": _DEFAULT_FONT_URL,
        }
      ]

    output: list[dict[str, object]] = []
    for item in value:
      if not isinstance(item, dict):
        continue
      full_name = item.get("full_name")
      if not isinstance(full_name, str) or not full_name.strip():
        full_name = font_name

      style = item.get("style")
      if not isinstance(style, str) or not style.strip():
        style = _DEFAULT_FONT_STYLE

      weight = item.get("weight")
      if isinstance(weight, bool):
        weight = _DEFAULT_FONT_WEIGHT
      elif isinstance(weight, int):
        pass
      elif isinstance(weight, float):
        weight = int(round(weight))
      elif isinstance(weight, str):
        try:
          weight = int(weight.strip())
        except ValueError:
          weight = _DEFAULT_FONT_WEIGHT
      else:
        weight = _DEFAULT_FONT_WEIGHT

      url = item.get("url")
      if not isinstance(url, str):
        url = _DEFAULT_FONT_URL

      output.append(
        {
          "full_name": full_name,
          "style": style,
          "weight": weight,
          "url": url,
        }
      )

    if output:
      return output
    return [
      {
        "full_name": font_name,
        "style": _DEFAULT_FONT_STYLE,
        "weight": _DEFAULT_FONT_WEIGHT,
        "url": _DEFAULT_FONT_URL,
      }
    ]

  processed = bundle.preprocess(image)
  tensor = bundle.tensor_transform(processed).unsqueeze(0).to(bundle.device)
  with torch.inference_mode():
    logits = bundle.model(tensor)
    probs = torch.softmax(logits, dim=1)
    k = min(max(top_k, 1), probs.shape[1])
    top_probs, top_indices = torch.topk(probs, k, dim=1)

  predictions = []
  for rank in range(k):
    pred_idx = int(top_indices[0, rank].item())
    pred_label = bundle.idx_to_class.get(pred_idx, f"unknown_{pred_idx}")
    pred_prob = round(float(top_probs[0, rank].item()), 2)
    raw_fonts = bundle.font_metadata_lookup.get(pred_label, [])
    fonts = coerce_font_entries(pred_label, raw_fonts)
    predictions.append(
      {
        "name": pred_label,
        "score": pred_prob,
        "fonts": fonts,
      }
    )
  return predictions


def run_inference_from_bytes(
  image_bytes: bytes,
  model_path: Path,
  top_k: int = DEFAULT_TOP_K,
  pad_ratio: float = DEFAULT_PAD_RATIO,
  min_word_len: int = DEFAULT_MIN_WORD_LEN,
  min_conf: float = DEFAULT_MIN_CONF,
  input_image_url: str | None = None,
  debug_dir: str | Path | None = None,
) -> dict[str, Any]:
  if top_k < 1:
    raise ValueError("top_k must be >= 1")

  image = load_image_from_bytes(image_bytes)
  image_width, image_height = image.size
  word_image = image
  word_box: WordBox | None = None
  try:
    word_image, padded_word_box = extract_largest_word(
      image,
      pad_ratio=pad_ratio,
      min_word_len=min_word_len,
      min_conf=min_conf,
    )
    word_box = map_box_to_original_image(
      box=padded_word_box,
      image_width=image_width,
      image_height=image_height,
      pad_ratio=pad_ratio,
    )
  except ValueError:
    # OCR is a best-effort crop. If no confident word is found,
    # keep the original image and continue inference.
    pass
  write_debug_images(
    debug_dir=debug_dir,
    original_image=image,
    word_box=word_box,
    model_input_image=word_image,
  )
  bundle = get_model_bundle(model_path)
  font_matches = run_model(word_image, bundle, top_k=top_k)
  return {
    "word": word_box.text if word_box else None,
    "word_box": (
      {
        "left": word_box.left,
        "top": word_box.top,
        "width": word_box.width,
        "height": word_box.height,
      }
      if word_box
      else None
    ),
    "input_image": {
      "width": image_width,
      "height": image_height,
      "image_url": input_image_url,
    },
    "font_matches": font_matches,
  }


def run_inference_from_url(
  image_url: str,
  model_path: Path,
  top_k: int = DEFAULT_TOP_K,
  pad_ratio: float = DEFAULT_PAD_RATIO,
  min_word_len: int = DEFAULT_MIN_WORD_LEN,
  min_conf: float = DEFAULT_MIN_CONF,
  debug_dir: str | Path | None = None,
) -> dict[str, Any]:
  image_bytes = download_image_bytes(image_url=image_url)
  return run_inference_from_bytes(
    image_bytes=image_bytes,
    model_path=model_path,
    top_k=top_k,
    pad_ratio=pad_ratio,
    min_word_len=min_word_len,
    min_conf=min_conf,
    input_image_url=image_url,
    debug_dir=debug_dir,
  )
