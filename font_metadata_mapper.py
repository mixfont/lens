"""
Map predicted font labels to enriched entries from `font_metadata.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_QUOTED_FIELD_RE = re.compile(r'^([a-z_]+):\s*"([^"]*)"')
_WEIGHT_FIELD_RE = re.compile(r"^weight:\s*(-?\d+)$")
_MIXFONT_STATIC_PREFIX = "https://static.mixfont.com/"
_DEFAULT_STYLE = "normal"
_DEFAULT_WEIGHT = 400


def _string_list(value: object) -> list[str]:
  if not isinstance(value, list):
    return []
  return [item for item in value if isinstance(item, str) and item]


def _dedupe(items: list[str]) -> list[str]:
  seen: set[str] = set()
  output: list[str] = []
  for item in items:
    if item in seen:
      continue
    seen.add(item)
    output.append(item)
  return output


def _normalize_path(path_value: str) -> str:
  return path_value.replace("\\", "/")


def _trim_to_fonts_path(path_value: str) -> str:
  normalized = _normalize_path(path_value).strip()
  if not normalized:
    return ""
  lowered = normalized.casefold()
  if lowered.startswith("fonts/"):
    return normalized
  marker = "/fonts/"
  marker_index = lowered.find(marker)
  if marker_index >= 0:
    return normalized[marker_index + 1:]
  return normalized


def _as_mixfont_static_url(path_value: str) -> str:
  normalized = _normalize_path(path_value).strip()
  if not normalized:
    return ""
  lowered = normalized.casefold()
  if lowered.startswith(_MIXFONT_STATIC_PREFIX.casefold()):
    return normalized
  if lowered.startswith("http://") or lowered.startswith("https://"):
    path_only = _trim_to_fonts_path(normalized)
  else:
    path_only = _trim_to_fonts_path(normalized)
  if not path_only:
    return ""
  return f"{_MIXFONT_STATIC_PREFIX}{path_only.lstrip('/')}"


def _is_absolute_path(path_value: str) -> bool:
  normalized = _normalize_path(path_value)
  return normalized.startswith("/") or bool(_WINDOWS_ABS_PATH_RE.match(normalized))


def _font_family_slug(font_name: str) -> str:
  return _NON_ALNUM_RE.sub("", font_name.casefold())


def _to_font_url(font_name: str, file_name: str) -> str:
  normalized_file = _normalize_path(file_name).strip()
  if not normalized_file:
    return ""
  if _is_absolute_path(normalized_file):
    return _as_mixfont_static_url(normalized_file)
  if normalized_file.casefold().startswith("fonts/"):
    return _as_mixfont_static_url(normalized_file)
  family_slug = _font_family_slug(font_name)
  if not family_slug:
    return _as_mixfont_static_url(normalized_file)
  return _as_mixfont_static_url(f"fonts/{family_slug}/{normalized_file}")


def _is_variable_filename(file_name: str) -> bool:
  lowered = file_name.casefold()
  if "[" in file_name and "]" in file_name:
    return True
  if "/variable/" in lowered or lowered.startswith("variable/"):
    return True
  return False


def _is_variable_font(entry: dict[str, object], file_names: list[str]) -> bool:
  metadata_pb = entry.get("metadata_pb")
  if isinstance(metadata_pb, str) and "axes {" in metadata_pb:
    return True
  for file_name in file_names:
    lowered = file_name.casefold()
    if "[" in file_name and "]" in file_name:
      return True
    if "/variable/" in lowered or lowered.startswith("variable/"):
      return True
  return False


def _extract_font_blocks(metadata_pb: str) -> list[list[str]]:
  blocks: list[list[str]] = []
  collecting = False
  block_depth = 0
  current_block: list[str] = []

  for raw_line in metadata_pb.splitlines():
    line = raw_line.strip()
    if not collecting:
      if line == "fonts {":
        collecting = True
        block_depth = 1
        current_block = []
      continue

    block_depth += line.count("{")
    block_depth -= line.count("}")
    if block_depth <= 0:
      blocks.append(current_block)
      collecting = False
      block_depth = 0
      current_block = []
      continue
    current_block.append(line)

  return blocks


def _parse_font_block(block_lines: list[str]) -> dict[str, object]:
  parsed: dict[str, object] = {}
  for line in block_lines:
    quoted_match = _QUOTED_FIELD_RE.match(line)
    if quoted_match:
      key, value = quoted_match.groups()
      if key in {"filename", "full_name", "style"} and key not in parsed:
        parsed[key] = value
      continue
    if "weight" in parsed:
      continue
    weight_match = _WEIGHT_FIELD_RE.match(line)
    if weight_match:
      try:
        parsed["weight"] = int(weight_match.group(1))
      except ValueError:
        pass
  return parsed


def _infer_style(file_name: str) -> str:
  lowered = file_name.casefold()
  if "italic" in lowered:
    return "italic"
  return "normal"


def _build_static_full_name(
  font_name: str,
  style: str,
  weight: object,
) -> str:
  if style == "italic":
    return f"{font_name} Italic"
  if isinstance(weight, int) and weight != 400:
    return f"{font_name} {weight}"
  return f"{font_name} Regular"


def _normalize_weight(value: object) -> int:
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(round(value))
  if isinstance(value, str):
    stripped = value.strip()
    if not stripped:
      return _DEFAULT_WEIGHT
    try:
      return int(stripped)
    except ValueError:
      return _DEFAULT_WEIGHT
  return _DEFAULT_WEIGHT


def _normalize_response_font(
  font_name: str,
  payload: dict[str, object],
) -> dict[str, object]:
  full_name = payload.get("full_name")
  if not isinstance(full_name, str) or not full_name.strip():
    full_name = font_name

  style = payload.get("style")
  if not isinstance(style, str) or not style.strip():
    style = _DEFAULT_STYLE

  url = payload.get("url")
  if not isinstance(url, str):
    url = ""

  return {
    "full_name": full_name,
    "style": style,
    "weight": _normalize_weight(payload.get("weight")),
    "url": url,
  }


def _collect_metadata_records(entry: dict[str, object]) -> list[dict[str, object]]:
  metadata_pb = entry.get("metadata_pb")
  if not isinstance(metadata_pb, str) or not metadata_pb:
    return []

  records: list[dict[str, object]] = []
  for block in _extract_font_blocks(metadata_pb):
    parsed = _parse_font_block(block)
    file_name = parsed.get("filename")
    if isinstance(file_name, str) and file_name:
      records.append(parsed)
  return records


def _collect_source_files(entry: dict[str, object]) -> list[str]:
  return _string_list(entry.get("font_files"))


def _normalize_font_entries(
  font_name: str,
  entry: dict[str, object],
) -> list[dict[str, object]]:
  records = _collect_metadata_records(entry)
  source_files = _collect_source_files(entry)

  files_from_records: list[str] = []
  for record in records:
    file_name = record.get("filename")
    if isinstance(file_name, str) and file_name:
      files_from_records.append(file_name)
  all_files = _dedupe(files_from_records + source_files)
  is_variable = _is_variable_font(entry, all_files)

  response_fonts: list[dict[str, object]] = []
  seen_keys: set[str] = set()

  def add_response_font(payload: dict[str, object]) -> None:
    normalized = _normalize_response_font(font_name, payload)
    url = normalized["url"]
    if isinstance(url, str) and url:
      dedupe_key = f"url::{url}"
    else:
      dedupe_key = (
        "fallback::"
        f"{normalized['full_name']}::{normalized['style']}::{normalized['weight']}"
      )
    if dedupe_key in seen_keys:
      return
    seen_keys.add(dedupe_key)
    response_fonts.append(normalized)

  for record in records:
    file_name = record.get("filename")
    url = (
      _to_font_url(font_name, file_name)
      if isinstance(file_name, str) and file_name
      else ""
    )

    full_name = record.get("full_name")
    if not isinstance(full_name, str) or not full_name:
      if is_variable:
        full_name = f"{font_name} Variable"
      else:
        full_name = _build_static_full_name(
          font_name=font_name,
          style=(
            record.get("style")
            if isinstance(record.get("style"), str)
            else _infer_style(file_name)
          ),
          weight=record.get("weight"),
        )

    if is_variable:
      add_response_font(
        {
          "full_name": full_name,
          "style": "variable",
          "url": url,
        }
      )
    else:
      style = (
        record.get("style")
        if isinstance(record.get("style"), str)
        else _infer_style(file_name)
      )
      add_response_font(
        {
          "full_name": full_name,
          "style": style,
          "url": url,
          "weight": record.get("weight"),
        }
      )

  for file_name in source_files:
    url = _to_font_url(font_name, file_name)
    if is_variable or _is_variable_filename(file_name):
      add_response_font(
        {
          "full_name": f"{font_name} Variable",
          "style": "variable",
          "url": url,
        }
      )
    else:
      style = _infer_style(file_name)
      add_response_font(
        {
          "full_name": _build_static_full_name(font_name, style, None),
          "style": style,
          "url": url,
        }
      )

  if not response_fonts:
    add_response_font({})

  return response_fonts


def load_font_metadata_lookup(model_path: Path) -> dict[str, list[dict[str, object]]]:
  metadata_path = model_path.parent / "font_metadata.json"
  if not metadata_path.exists():
    return {}

  try:
    with metadata_path.open("r", encoding="utf-8") as handle:
      payload = json.load(handle)
  except (OSError, json.JSONDecodeError):
    return {}

  if not isinstance(payload, dict):
    return {}
  fonts = payload.get("fonts")
  if not isinstance(fonts, dict):
    return {}

  lookup: dict[str, list[dict[str, object]]] = {}
  for font_name, raw_entry in fonts.items():
    if not isinstance(font_name, str) or not isinstance(raw_entry, dict):
      continue
    lookup[font_name] = _normalize_font_entries(font_name, raw_entry)
  return lookup
