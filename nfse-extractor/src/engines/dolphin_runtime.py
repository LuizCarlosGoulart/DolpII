"""ByteDance Dolphin-v2 runtime factory.

Provides ``load_dolphin_runtime``, a callable that loads the model and returns
a predictor compatible with ``DolphinExtractionAdapter``.

Usage in notebook 03::

    CONFIG['runtime_factory_path'] = 'src.engines.dolphin_runtime.load_dolphin_runtime'
    CONFIG['model_path'] = 'ByteDance/Dolphin'   # or a local HF-snapshot path
    CONFIG['device']     = None                  # None = auto (cuda if available)

The predictor implements the two-stage Dolphin inference pipeline
(layout/reading-order parse → element recognition) and converts every
bounding box from Dolphin's native xyxy format to the project's xywh
convention expected by ``ConfigDrivenOutputNormalizer``.

Utility helpers (``_resize_img``, ``_parse_layout_string``,
``_process_coordinates``, ``_check_bbox_overlap``) are inlined from
``bytedance/Dolphin`` (MIT licence) so no extra repo clone is required.
"""

from __future__ import annotations

import re
from typing import Any, Callable


def load_dolphin_runtime(
    *,
    model_path: str | None = None,
    device: str | None = None,
    max_batch_size: int = 4,
) -> Callable[[Any], list[dict]]:
    """Load Dolphin-v2 and return a predictor callable.

    The returned predictor accepts a single PIL ``Image`` and returns a list
    of element dicts::

        {
            "label":         str,          # e.g. "para", "tab", "equ"
            "text":          str,          # recognised text / LaTeX / HTML
            "bbox":          [x, y, w, h], # xywh, original-image pixels
            "reading_order": int,
            "tags":          list[str],
        }

    ``bbox`` is in **xywh** format (x-origin, y-origin, width, height) to
    match the ``bounding_box`` convention used by the rest of the project.

    Args:
        model_path:      HuggingFace model ID or local snapshot path.
                         Defaults to ``'ByteDance/Dolphin-v2'``.
        device:          ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
        max_batch_size:  Maximum elements per VLM inference call (default 4).
                         Reduce if you hit OOM on small-VRAM GPUs.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info, smart_resize  # noqa: F401

    resolved_path = model_path or "ByteDance/Dolphin-v2"

    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[dolphin_runtime] Loading model from {resolved_path!r} on {device!r} …")
    processor = AutoProcessor.from_pretrained(resolved_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(resolved_path)
    model.eval()
    model.to(device)
    if device == "cuda":
        model = model.bfloat16()
    else:
        model = model.float()
    processor.tokenizer.padding_side = "left"
    print("[dolphin_runtime] Model loaded and ready.")

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _chat(prompts: list[str], images: list[Any]) -> list[str]:
        """Run a batched VLM forward pass. Returns one string per pair."""
        from qwen_vl_utils import process_vision_info as _pvi

        processed = [_resize_img(img) for img in images]
        all_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": q},
                    ],
                }
            ]
            for img, q in zip(processed, prompts)
        ]
        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]
        all_image_inputs: list[Any] = []
        for msgs in all_messages:
            img_inputs, _ = _pvi(msgs)
            all_image_inputs.extend(img_inputs)

        inputs = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        if device == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _chat_batched(prompts: list[str], images: list[Any]) -> list[str]:
        """Chunk ``_chat`` calls so each batch stays within ``max_batch_size``."""
        results: list[str] = []
        for i in range(0, len(images), max_batch_size):
            results.extend(_chat(prompts[i : i + max_batch_size], images[i : i + max_batch_size]))
        return results

    # ── Predictor (returned to the caller) ────────────────────────────────────

    def predictor(image: Any) -> list[dict]:
        """Full two-stage Dolphin inference on a single PIL Image.

        Returns element dicts with ``bbox`` in **xywh** format.
        """
        from qwen_vl_utils import smart_resize as _sr

        # Stage 1 ── reading order / layout
        layout_output = _chat(["Parse the reading order of this document."], [image])[0]

        # Free VRAM between stage 1 and stage 2
        if device == "cuda":
            torch.cuda.empty_cache()

        # Parse stage-1 output into (coords, label, tags) triples
        layout_list = _parse_layout_string(layout_output)

        # Fall back to full-page "distorted" mode when layout parse fails
        img_w, img_h = image.size
        if not layout_list or not (
            layout_output.startswith("[") and layout_output.endswith("]")
        ):
            layout_list = [([0, 0, img_w, img_h], "distorted_page", [])]
        elif len(layout_list) > 1 and _check_bbox_overlap(layout_list, image, _sr):
            print(
                "[dolphin_runtime] High bbox overlap detected; "
                "falling back to distorted_page mode."
            )
            layout_list = [([0, 0, img_w, img_h], "distorted_page", [])]

        # Stage 2 ── element recognition
        tab_elems: list[dict] = []
        equ_elems: list[dict] = []
        code_elems: list[dict] = []
        text_elems: list[dict] = []
        figure_results: list[dict] = []
        reading_order = 0

        for bbox, label, tags in layout_list:
            try:
                if label == "distorted_page":
                    x1, y1 = 0, 0
                    x2, y2 = image.size
                    pil_crop = image
                else:
                    x1, y1, x2, y2 = _process_coordinates(bbox, image, _sr)
                    pil_crop = image.crop((x1, y1, x2, y2))

                if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                    if label == "fig":
                        figure_results.append(
                            {
                                "label": label,
                                "text": "",
                                # bbox already in original-image xyxy → convert to xywh
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "reading_order": reading_order,
                                "tags": tags,
                                "confidence": 0.92,
                            }
                        )
                    else:
                        element_info = {
                            "crop": pil_crop,
                            "label": label,
                            "bbox": [x1, y1, x2, y2],  # kept as xyxy until text is added
                            "reading_order": reading_order,
                            "tags": tags,
                        }
                        if label == "tab":
                            tab_elems.append(element_info)
                        elif label == "equ":
                            equ_elems.append(element_info)
                        elif label == "code":
                            code_elems.append(element_info)
                        else:
                            text_elems.append(element_info)

                reading_order += 1

            except Exception as exc:  # noqa: BLE001
                print(f"[dolphin_runtime] Skipped element label={label!r}: {exc}")
                continue

        recognition_results = list(figure_results)

        for elems, prompt in (
            (tab_elems, "Parse the table in the image."),
            (equ_elems, "Read formula in the image."),
            (code_elems, "Read code in the image."),
            (text_elems, "Read text in the image."),
        ):
            if not elems:
                continue
            crops = [e["crop"] for e in elems]
            texts = _chat_batched([prompt] * len(crops), crops)
            for elem, text in zip(elems, texts):
                x1, y1, x2, y2 = elem["bbox"]
                recognition_results.append(
                    {
                        "label": elem["label"],
                        "text": text.strip(),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # xyxy → xywh
                        "reading_order": elem["reading_order"],
                        "tags": elem["tags"],
                        "confidence": 0.92,
                    }
                )

        recognition_results.sort(key=lambda x: x.get("reading_order", 0))

        # Decompose HTML table elements into per-row text elements so the
        # normalizer receives individual lines instead of multi-KB HTML blobs.
        expanded: list[dict] = []
        for elem in recognition_results:
            if elem.get("label") == "tab" and "<table" in elem.get("text", "").lower():
                expanded.extend(_decompose_table_rows(elem))
            else:
                expanded.append(elem)
        return expanded

    return predictor


# ─── Utility helpers (inlined from bytedance/Dolphin utils/utils.py, MIT) ─────


def _resize_img(image: Any, max_size: int = 896, min_size: int = 28) -> Any:
    """Resize so the longest side ≤ max_size and the shortest side ≥ min_size.

    896 matches the Dolphin demo's PDF page target size and keeps VRAM usage
    within T4 (15 GB) limits.  The original demo used 1600 but that requires
    ~4-5x more VRAM for the visual attention pass on full-page images.
    """
    width, height = image.size
    if max(width, height) <= max_size and min(width, height) >= min_size:
        return image
    if max(width, height) > max_size:
        if width >= height:
            new_w, new_h = max_size, int(height * max_size / width)
        else:
            new_w, new_h = int(width * max_size / height), max_size
        image = image.resize((new_w, new_h))
        width, height = image.size
    if min(width, height) < min_size:
        if width <= height:
            new_w, new_h = min_size, int(height * min_size / width)
        else:
            new_w, new_h = int(width * min_size / height), min_size
        image = image.resize((new_w, new_h))
    return image


def _parse_layout_string(bbox_str: str) -> list[tuple]:
    """Parse a Dolphin stage-1 layout string into ``(coords, label, tags)`` tuples.

    Supports both the original ``[x1,y1,x2,y2] label`` format and the newer
    ``[x1,y1,x2,y2][label][tag…][PAIR_SEP]`` format.
    """
    parsed: list[tuple] = []
    segments: list[str] = []
    for part in bbox_str.split("[PAIR_SEP]"):
        segments.extend(part.split("[RELATION_SEP]"))

    coord_re = re.compile(r"\[(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+)\]")
    non_coord_re = re.compile(r"\[([^\]]+)\]")

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        coord_match = coord_re.search(segment)
        label_candidates = [
            m
            for m in non_coord_re.findall(segment)
            if not re.fullmatch(r"\d*\.?\d+,\d*\.?\d+,\d*\.?\d+,\d*\.?\d+", m)
        ]
        if coord_match and label_candidates:
            coords = [float(coord_match.group(i)) for i in range(1, 5)]
            label = label_candidates[0].strip()
            tags = label_candidates[1:]
            parsed.append((coords, label, tags))

    return parsed


def _process_coordinates(
    coords: list[float],
    pil_image: Any,
    smart_resize: Any,
) -> tuple[int, int, int, int]:
    """Map model-space coordinates back to original-image pixel coordinates."""
    import numpy as np

    original_w, original_h = pil_image.size
    resized_pil = _resize_img(pil_image)
    resized_arr = np.array(resized_pil)
    rh, rw = resized_arr.shape[:2]
    rh, rw = smart_resize(rh, rw, factor=28, min_pixels=784, max_pixels=2_560_000)

    w_ratio = original_w / rw
    h_ratio = original_h / rh
    x1 = max(0, min(int(coords[0] * w_ratio), original_w - 1))
    y1 = max(0, min(int(coords[1] * h_ratio), original_h - 1))
    x2 = max(x1 + 1, min(int(coords[2] * w_ratio), original_w))
    y2 = max(y1 + 1, min(int(coords[3] * h_ratio), original_h))
    return x1, y1, x2, y2


def _check_bbox_overlap(
    layout_list: list[tuple],
    image: Any,
    smart_resize: Any,
    iou_threshold: float = 0.1,
    overlap_ratio_threshold: float = 0.25,
) -> bool:
    """Return ``True`` if excessive bbox overlap suggests a distorted scan.

    Uses a vectorised IoU matrix; mirrors the logic in the official demo.
    """
    import numpy as np

    if len(layout_list) <= 1:
        return False

    boxes = np.array(
        [list(_process_coordinates(bbox, image, smart_resize)) for bbox, _, _ in layout_list],
        dtype=float,
    )
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = areas[:, None] + areas[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)
    has_overlap = (iou > iou_threshold).any(axis=1)
    return bool(has_overlap.mean() > overlap_ratio_threshold)


def _decompose_table_rows(element: dict) -> list[dict]:
    """Decompose an HTML ``<table>`` element into one dict per table row.

    Each row becomes a plain-text ``"para"`` element whose text is the
    concatenation of its cell contents separated by double spaces.  This
    converts Dolphin's document-level HTML blob into the short text lines
    that ``ConfigDrivenOutputNormalizer`` expects.

    Falls back to returning the original element unchanged when no ``<tr>``
    rows can be found (e.g. malformed HTML or plain-text table output).
    """
    import re as _re

    html_text = element.get("text", "")
    row_re = _re.compile(r"<tr[^>]*>(.*?)</tr>", _re.DOTALL | _re.IGNORECASE)
    cell_re = _re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", _re.DOTALL | _re.IGNORECASE)
    tag_re = _re.compile(r"<[^>]+>")

    all_rows = list(row_re.finditer(html_text))
    if not all_rows:
        return [element]

    n_rows = len(all_rows)
    parent_bbox = element.get("bbox")

    rows: list[dict] = []
    for row_idx, row_match in enumerate(all_rows):
        cell_texts: list[str] = []
        for cell_match in cell_re.finditer(row_match.group(1)):
            text = tag_re.sub(" ", cell_match.group(1))
            text = _re.sub(r"\s+", " ", text).strip()
            if text:
                cell_texts.append(text)
        row_text = "  ".join(cell_texts)
        if not row_text.strip():
            continue

        row_bbox = None
        if parent_bbox is not None:
            x, y, w, h = parent_bbox
            row_h = h / n_rows
            row_bbox = [x, y + row_idx * row_h, w, row_h]

        row_dict = {
            "label": "para",
            "text": row_text,
            "bbox": row_bbox,
            "reading_order": element["reading_order"] * 1000 + row_idx,
            "tags": element.get("tags", []),
            "confidence": element.get("confidence", 0.92),
        }
        # Split each row into sub-elements (one per cell/label:value pair) so the
        # normalizer can match labels against their adjacent value elements.
        rows.extend(_split_row_into_subelements(row_dict))

    return rows if rows else [element]


def _split_row_into_subelements(row_dict: dict) -> list[dict]:
    """Split a merged row element into fine-grained sub-elements.

    Two-pass splitting strategy:

    1. **Double-space** split — separates distinct table cells joined with
       ``"  "`` (the separator used by :func:`_decompose_table_rows`).
    2. **Colon split** — each cell of the form ``"Label: Value"`` is further
       split into two sub-elements so the normalizer can pair the label token
       with the value element that follows it (the same structure produced by
       Tesseract's word-level output).

    Sub-elements inherit all fields from ``row_dict`` and receive proportional
    X-offset bounding boxes (same Y, scaled widths based on character count).
    When only one sub-element results, the original dict is returned unchanged.
    """
    text = row_dict.get("text", "")
    bbox = row_dict.get("bbox")  # [x, y, w, h] or None

    # ── Pass 1: split on double-space (cell boundaries) ──────────────────────
    segments: list[str] = [s for s in text.split("  ") if s.strip()]
    if not segments:
        return [row_dict]

    # ── Pass 2: split each segment on ": " OR typed-value boundary ─────────
    # When a cell has an explicit colon-separator ("Label: Value") we use it.
    # Otherwise we look for a typed-value token (date, document number, money,
    # long number) to infer where the label text ends and the value begins —
    # e.g. "Número da Nota 00032794" → ["Número da Nota", "00032794"].
    sub_texts: list[str] = []
    for segment in segments:
        if ": " in segment:
            label_part, value_part = segment.split(": ", 1)
            label_part = label_part.strip()
            value_part = value_part.strip()
            if label_part:
                sub_texts.append(label_part)
            if value_part:
                sub_texts.append(value_part)
        else:
            sub_texts.extend(_typed_value_split(segment))

    sub_texts = [s for s in sub_texts if s]
    if len(sub_texts) <= 1:
        return [row_dict]

    # ── Assign proportional X-offset bounding boxes ──────────────────────────
    total_chars = sum(len(s) for s in sub_texts)
    if total_chars == 0:
        return [row_dict]

    result: list[dict] = []
    x_origin = float(bbox[0]) if bbox else 0.0
    row_y = float(bbox[1]) if bbox else 0.0
    row_w = float(bbox[2]) if bbox else 0.0
    row_h = float(bbox[3]) if bbox else 0.0
    x_cursor = x_origin

    for sub_text in sub_texts:
        sub_w = row_w * len(sub_text) / total_chars
        sub_elem = dict(row_dict)
        sub_elem["text"] = sub_text
        if bbox is not None:
            sub_elem["bbox"] = [x_cursor, row_y, sub_w, row_h]
        result.append(sub_elem)
        x_cursor += sub_w

    return result


# Matches the start of a "typed value" token within a mixed label+value string.
# Used by ``_typed_value_split`` to detect where label text ends and a value
# begins when no explicit ": " separator is present.
#
# Pattern ordering matters: more-specific patterns (CNPJ, CPF, formatted money,
# full date) are listed before the catch-all long-integer pattern so that the
# alternation chooses the semantically correct match when multiple alternatives
# could match at the same position.
_TYPED_VALUE_BOUNDARY_RE = re.compile(
    r"(?:"
    r"\b\d{2}/\d{2}/\d{4}\b"                    # DD/MM/YYYY date
    r"|\b\d{2}/\d{4}\b"                          # MM/YYYY competence date
    r"|\b\d{2}[\.,]\d{3}\.\d{3}/\d{4}-\d{2}\b"  # CNPJ with mask
    r"|\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"           # CPF with mask
    r"|R\$\s*\d"                                 # currency starting with R$
    r"|\b\d{1,3}(?:\.\d{3})+,\d{2}\b"          # formatted money  (1.234,56)
    r"|\b\d{5}-\d{3}\b"                         # CEP (Brazilian ZIP code)
    r"|\b\d{4,}\b"                              # any integer with ≥ 4 digits
    r")"
)


def _typed_value_split(segment: str) -> list[str]:
    """Split a mixed label+value segment at the first typed-value boundary.

    Examples::

        "Número da Nota 00032794"  →  ["Número da Nota", "00032794"]
        "Emissão 05/09/2021"       →  ["Emissão", "05/09/2021"]
        "CNPJ 12.345.678/0001-90"  →  ["CNPJ", "12.345.678/0001-90"]
        "05/09/2021"               →  ["05/09/2021"]        (pure value, no split)
        "Série E"                  →  ["Série E"]           (no typed token found)

    When no typed-value token is found, or the token starts at position 0 (the
    segment is already a pure value), the original segment is returned unchanged
    inside a one-element list.
    """
    match = _TYPED_VALUE_BOUNDARY_RE.search(segment)
    if match is None or match.start() == 0:
        return [segment]
    prefix = segment[: match.start()].strip()
    # Only split when the prefix contains at least one letter — guards against
    # splitting on e.g. "  00032794" where the "prefix" would be just spaces.
    if not prefix or not any(c.isalpha() for c in prefix):
        return [segment]
    suffix = segment[match.start() :].strip()
    return [p for p in [prefix, suffix] if p]
