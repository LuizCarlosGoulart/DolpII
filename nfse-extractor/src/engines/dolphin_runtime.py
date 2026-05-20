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
                    }
                )

        recognition_results.sort(key=lambda x: x.get("reading_order", 0))
        return recognition_results

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
