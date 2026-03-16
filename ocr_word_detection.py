"""
OCR word-box detection helpers for Lens inference.

Call `extract_largest_word(...)` with a PIL image to locate the most useful
single-word crop for font classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from PIL import Image


PRIMARY_PSM = 6
FALLBACK_PSMS = (11, 3)
UPSCALE_FACTOR = 2.0
PAD_RATIO_MULTIPLIER = 2.0
WORD_CROP_PADDING_PX = 3


@dataclass(frozen=True)
class WordBox:
  text: str
  conf: float
  left: int
  top: int
  width: int
  height: int

  @property
  def right(self) -> int:
    return self.left + self.width

  @property
  def bottom(self) -> int:
    return self.top + self.height


def _coerce_float(value: Any, default: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _has_alphanumeric(text: str) -> bool:
  return any(char.isalnum() for char in text)


def _box_score(box: WordBox) -> tuple[int, int, float, int]:
  # Prefer OCR tokens with at least one alphanumeric character.
  return (
    1 if _has_alphanumeric(box.text) else 0,
    box.width * box.height,
    max(box.conf, 0.0),
    len(box.text),
  )


def _build_relaxed_thresholds(
  min_word_len: int,
  min_conf: float,
) -> list[tuple[int, float]]:
  """
  Build increasingly permissive OCR filter thresholds.

  Order:
  1) Keep requested minimum length and relax confidence.
  2) Reduce minimum length and retry the same confidence ladder.
  """
  base_len = max(1, int(min_word_len))
  lengths = [base_len]
  if base_len > 2:
    lengths.append(2)
  if base_len > 1:
    lengths.append(1)

  confs = [float(min_conf)]
  if min_conf > 0.0:
    confs.append(0.0)
  if min_conf > -1.0:
    confs.append(-1.0)

  ordered: list[tuple[int, float]] = []
  seen: set[tuple[int, float]] = set()
  for length in lengths:
    for conf in confs:
      key = (length, conf)
      if key in seen:
        continue
      seen.add(key)
      ordered.append(key)
  return ordered


def _pixel_to_gray(value: Any) -> int:
  if isinstance(value, tuple):
    if not value:
      return 255
    channels = value[:3]
    return int(round(sum(channels) / len(channels)))
  try:
    return int(value)
  except (TypeError, ValueError):
    return 255


def _sample_edge_average_gray(image: "Image.Image") -> int:
  width, height = image.size
  if width == 0 or height == 0:
    return 255

  pixels = image.load()
  step_x = max(1, width // 50)
  step_y = max(1, height // 50)
  total = 0
  count = 0

  for x in range(0, width, step_x):
    total += _pixel_to_gray(pixels[x, 0])
    count += 1
    if height > 1:
      total += _pixel_to_gray(pixels[x, height - 1])
      count += 1

  for y in range(0, height, step_y):
    total += _pixel_to_gray(pixels[0, y])
    count += 1
    if width > 1:
      total += _pixel_to_gray(pixels[width - 1, y])
      count += 1

  if count == 0:
    return 255
  return int(round(total / count))


def pad_image_for_ocr(
  image: "Image.Image",
  pad_ratio: float,
  pad_multiplier: float = PAD_RATIO_MULTIPLIER,
) -> "Image.Image":
  try:
    from PIL import ImageOps
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pillow to run image inference."
    ) from exc

  width, height = image.size
  pad_x, pad_y = compute_ocr_padding(
    image_width=width,
    image_height=height,
    pad_ratio=pad_ratio,
    pad_multiplier=pad_multiplier,
  )
  if pad_x == 0 and pad_y == 0:
    return image

  edge_average = _sample_edge_average_gray(image)
  return ImageOps.expand(
    image,
    border=(pad_x, pad_y, pad_x, pad_y),
    fill=edge_average,
  )


def compute_ocr_padding(
  image_width: int,
  image_height: int,
  pad_ratio: float,
  pad_multiplier: float = PAD_RATIO_MULTIPLIER,
) -> tuple[int, int]:
  if image_width <= 0 or image_height <= 0:
    return 0, 0

  effective_ratio = pad_ratio * pad_multiplier
  pad_x = int(round(image_width * effective_ratio))
  pad_y = int(round(image_height * effective_ratio))
  if effective_ratio > 0:
    pad_x = max(1, pad_x)
    pad_y = max(1, pad_y)
  return pad_x, pad_y


def map_box_to_original_image(
  box: WordBox,
  image_width: int,
  image_height: int,
  pad_ratio: float,
  pad_multiplier: float = PAD_RATIO_MULTIPLIER,
) -> WordBox:
  """Translate a padded OCR box back into original-image coordinates."""
  if image_width <= 0 or image_height <= 0:
    return box

  pad_x, pad_y = compute_ocr_padding(
    image_width=image_width,
    image_height=image_height,
    pad_ratio=pad_ratio,
    pad_multiplier=pad_multiplier,
  )

  left = box.left - pad_x
  top = box.top - pad_y
  right = box.right - pad_x
  bottom = box.bottom - pad_y

  left = max(0, min(left, image_width - 1))
  top = max(0, min(top, image_height - 1))
  right = max(left + 1, min(right, image_width))
  bottom = max(top + 1, min(bottom, image_height))

  return WordBox(
    text=box.text,
    conf=box.conf,
    left=left,
    top=top,
    width=right - left,
    height=bottom - top,
  )


def find_largest_word_box(
  ocr_data: dict[str, list],
  min_word_len: int,
  min_conf: float,
) -> WordBox | None:
  best_box: WordBox | None = None
  best_score: tuple[int, int, float, int] | None = None

  texts = ocr_data.get("text", [])
  for i, text in enumerate(texts):
    cleaned = str(text).strip() if text is not None else ""
    if not cleaned or len(cleaned) < min_word_len:
      continue
    try:
      conf = _coerce_float(ocr_data.get("conf", [])[i], -1.0)
      left = int(ocr_data["left"][i])
      top = int(ocr_data["top"][i])
      width = int(ocr_data["width"][i])
      height = int(ocr_data["height"][i])
    except (IndexError, KeyError, TypeError, ValueError):
      continue
    if conf < min_conf:
      continue
    if width <= 0 or height <= 0:
      continue

    # Prefer alphanumeric text; fallback to non-alphanumeric only if needed.
    score = (
      1 if _has_alphanumeric(cleaned) else 0,
      width * height,
      max(conf, 0.0),
      len(cleaned),
    )
    if best_score is None or score > best_score:
      best_score = score
      best_box = WordBox(
        text=cleaned,
        conf=conf,
        left=left,
        top=top,
        width=width,
        height=height,
      )
  return best_box


def _collect_all_word_boxes(ocr_data: dict[str, list]) -> list[WordBox]:
  boxes: list[WordBox] = []
  texts = ocr_data.get("text", [])
  for i, text in enumerate(texts):
    cleaned = str(text).strip() if text is not None else ""
    if not cleaned:
      continue
    try:
      conf = _coerce_float(ocr_data.get("conf", [])[i], -1.0)
      left = int(ocr_data["left"][i])
      top = int(ocr_data["top"][i])
      width = int(ocr_data["width"][i])
      height = int(ocr_data["height"][i])
    except (IndexError, KeyError, TypeError, ValueError):
      continue
    if width <= 0 or height <= 0:
      continue
    boxes.append(
      WordBox(
        text=cleaned,
        conf=conf,
        left=left,
        top=top,
        width=width,
        height=height,
      )
    )
  return boxes


def _dedupe_boxes(boxes: list[WordBox]) -> list[WordBox]:
  unique: list[WordBox] = []
  seen: set[tuple[str, int, int, int, int]] = set()
  for box in boxes:
    key = (box.text.lower(), box.left, box.top, box.width, box.height)
    if key in seen:
      continue
    seen.add(key)
    unique.append(box)
  return unique


def _draw_boxes(
  image: "Image.Image",
  boxes: list[WordBox],
  highlight_box: WordBox | None,
) -> "Image.Image":
  try:
    from PIL import ImageDraw
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pillow to run image inference."
    ) from exc

  overlay = image.convert("RGB")
  draw = ImageDraw.Draw(overlay)
  line_width = max(1, int(round(min(overlay.size) * 0.004)))
  for box in boxes:
    draw.rectangle(
      (box.left, box.top, box.right, box.bottom),
      outline=(255, 64, 64),
      width=line_width,
    )

  if highlight_box:
    draw.rectangle(
      (
        highlight_box.left,
        highlight_box.top,
        highlight_box.right,
        highlight_box.bottom,
      ),
      outline=(64, 255, 96),
      width=max(2, line_width + 1),
    )
  return overlay


def draw_all_ocr_boxes(
  image: "Image.Image",
  pad_ratio: float,
  highlight_box: WordBox | None = None,
) -> "Image.Image":
  try:
    import pytesseract
    from pytesseract import Output
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pytesseract to run OCR inference."
    ) from exc

  gray = image.convert("L")
  padded = pad_image_for_ocr(gray, pad_ratio)
  all_boxes: list[WordBox] = []

  for psm in (PRIMARY_PSM, *FALLBACK_PSMS):
    ocr_data = pytesseract.image_to_data(
      padded,
      output_type=Output.DICT,
      config=f"--psm {psm}",
    )
    all_boxes.extend(_collect_all_word_boxes(ocr_data))

  upscaled = _resize_for_ocr(padded, UPSCALE_FACTOR)
  for psm in (PRIMARY_PSM, *FALLBACK_PSMS):
    ocr_data = pytesseract.image_to_data(
      upscaled,
      output_type=Output.DICT,
      config=f"--psm {psm}",
    )
    upscaled_boxes = _collect_all_word_boxes(ocr_data)
    all_boxes.extend(
      [
        _scale_box_to_base(
          box=box,
          scale=UPSCALE_FACTOR,
          base_width=padded.width,
          base_height=padded.height,
        )
        for box in upscaled_boxes
      ]
    )

  unique_boxes = _dedupe_boxes(all_boxes)
  return _draw_boxes(padded, unique_boxes, highlight_box)


def _run_ocr_for_psms(
  image: "Image.Image",
  min_word_len: int,
  min_conf: float,
  psms: tuple[int, ...],
  pytesseract: Any,
  output_type: Any,
) -> WordBox | None:
  best_box: WordBox | None = None
  for psm in psms:
    ocr_data = pytesseract.image_to_data(
      image,
      output_type=output_type,
      config=f"--psm {psm}",
    )
    candidate = find_largest_word_box(ocr_data, min_word_len, min_conf)
    if candidate is None:
      continue
    if best_box is None:
      best_box = candidate
      continue
    candidate_score = _box_score(candidate)
    best_score = _box_score(best_box)
    if candidate_score > best_score:
      best_box = candidate
  return best_box


def _resize_for_ocr(image: "Image.Image", scale: float) -> "Image.Image":
  try:
    from PIL import Image as PILImage
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pillow to run image inference."
    ) from exc

  if scale == 1.0:
    return image
  width = max(1, int(round(image.width * scale)))
  height = max(1, int(round(image.height * scale)))
  resampling = getattr(PILImage, "Resampling", PILImage)
  return image.resize((width, height), resampling.BICUBIC)


def _scale_box_to_base(
  box: WordBox,
  scale: float,
  base_width: int,
  base_height: int,
) -> WordBox:
  if scale == 1.0:
    return box

  left = max(0, int(round(box.left / scale)))
  top = max(0, int(round(box.top / scale)))
  right = max(left + 1, int(round(box.right / scale)))
  bottom = max(top + 1, int(round(box.bottom / scale)))

  right = min(base_width, right)
  bottom = min(base_height, bottom)
  left = min(left, max(0, right - 1))
  top = min(top, max(0, bottom - 1))

  return WordBox(
    text=box.text,
    conf=box.conf,
    left=left,
    top=top,
    width=right - left,
    height=bottom - top,
  )


def _crop_to_box(image: "Image.Image", box: WordBox) -> "Image.Image":
  width, height = image.size
  if width == 0 or height == 0:
    return image
  left = max(0, min(box.left, width - 1))
  top = max(0, min(box.top, height - 1))
  right = max(left + 1, min(box.right, width))
  bottom = max(top + 1, min(box.bottom, height))
  return image.crop((left, top, right, bottom))


def _expand_box_for_crop(
  box: WordBox,
  image_width: int,
  image_height: int,
  padding: int = WORD_CROP_PADDING_PX,
) -> WordBox:
  if image_width <= 0 or image_height <= 0:
    return box

  left = max(0, box.left - padding)
  top = max(0, box.top - padding)
  right = min(image_width, box.right + padding)
  bottom = min(image_height, box.bottom + padding)
  right = max(left + 1, right)
  bottom = max(top + 1, bottom)

  return WordBox(
    text=box.text,
    conf=box.conf,
    left=left,
    top=top,
    width=right - left,
    height=bottom - top,
  )


def extract_largest_word(
  image: "Image.Image",
  pad_ratio: float,
  min_word_len: int,
  min_conf: float,
) -> tuple["Image.Image", WordBox]:
  try:
    import pytesseract
    from pytesseract import Output
  except ImportError as exc:
    raise RuntimeError(
      "Missing dependency. Install pytesseract to run OCR inference."
    ) from exc

  gray = image.convert("L")
  padded = pad_image_for_ocr(gray, pad_ratio)

  upscaled = _resize_for_ocr(padded, UPSCALE_FACTOR)

  for effective_min_word_len, effective_min_conf in _build_relaxed_thresholds(
    min_word_len=min_word_len,
    min_conf=min_conf,
  ):
    stage_one_box = _run_ocr_for_psms(
      image=padded,
      min_word_len=effective_min_word_len,
      min_conf=effective_min_conf,
      psms=(PRIMARY_PSM,),
      pytesseract=pytesseract,
      output_type=Output.DICT,
    )

    stage_two_box = _run_ocr_for_psms(
      image=padded,
      min_word_len=effective_min_word_len,
      min_conf=effective_min_conf,
      psms=FALLBACK_PSMS,
      pytesseract=pytesseract,
      output_type=Output.DICT,
    )

    stage_three_box = _run_ocr_for_psms(
      image=upscaled,
      min_word_len=effective_min_word_len,
      min_conf=effective_min_conf,
      psms=(PRIMARY_PSM, *FALLBACK_PSMS),
      pytesseract=pytesseract,
      output_type=Output.DICT,
    )
    scaled_stage_three_box: WordBox | None = None
    if stage_three_box:
      scaled_stage_three_box = _scale_box_to_base(
        box=stage_three_box,
        scale=UPSCALE_FACTOR,
        base_width=padded.width,
        base_height=padded.height,
      )

    candidates = [stage_one_box, stage_two_box, scaled_stage_three_box]
    best_box = max(
      (box for box in candidates if box is not None),
      key=_box_score,
      default=None,
    )
    if best_box:
      padded_box = _expand_box_for_crop(
        box=best_box,
        image_width=padded.width,
        image_height=padded.height,
      )
      return _crop_to_box(padded, padded_box), padded_box

  raise ValueError("No OCR words detected in the image.")
