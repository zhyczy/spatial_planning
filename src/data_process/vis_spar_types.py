"""
Visualize one sample per question type from train_10k.json.
Saves annotated images to vis_results/spar_types/<type_name>/
"""
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
_SPATIAL_PLANNING_DIR = _THIS_DIR.parent.parent

_DATA_ROOT = _SPATIAL_PLANNING_DIR / "datasets" / "train" / "SPAR_7M" / "spar"
_SCANNET_ROOT = _DATA_ROOT / "scannet" / "images"
_SCANNETPP_ROOT = _DATA_ROOT / "scannetpp" / "images"
_STRUCTURED3D_ROOT = _DATA_ROOT / "structured3d" / "images"
_VIS_ROOT = _SPATIAL_PLANNING_DIR / "vis_results" / "spar_types"

_TRAIN_JSON = _DATA_ROOT / "train_10k.json"

POINT_RADIUS = 8
BBOX_WIDTH = 3

COLOR_MAP = {
    "red_point": (255, 60, 60),
    "green_point": (60, 220, 60),
    "blue_point": (60, 100, 255),
    "red_bbox": (255, 60, 60),
    "blue_bbox": (60, 100, 255),
    "green_bbox": (60, 220, 60),
}


def find_image(rel_path: str) -> Optional[Path]:
    """Search across known base directories."""
    for base in [_SCANNETPP_ROOT, _SCANNET_ROOT, _STRUCTURED3D_ROOT, _DATA_ROOT]:
        p = base / rel_path
        if p.exists():
            return p
    return None


def draw_annotations(img: Image.Image, entry: dict, img_idx: int) -> Image.Image:
    """Draw points and bboxes that belong to this image index."""
    draw = ImageDraw.Draw(img)

    for key, color in COLOR_MAP.items():
        coords_list = entry.get(key)
        if not coords_list:
            continue
        # Determine which image each annotation belongs to
        idx_key = key.replace("point", "img_idx").replace("bbox", "img_idx")
        point_img_idx = entry.get("point_img_idx", [])
        bbox_img_idx = entry.get("bbox_img_idx", [])

        if "point" in key:
            img_indices = point_img_idx
        else:
            img_indices = bbox_img_idx

        for i, coords in enumerate(coords_list):
            # img_indices is a flat list where position i corresponds to annotation i
            assigned_img = img_indices[i] if i < len(img_indices) else 0
            if isinstance(assigned_img, list):
                assigned_img = assigned_img[0]
            if assigned_img != img_idx:
                continue

            if "point" in key:
                x, y = coords
                r = POINT_RADIUS
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
            else:
                x1, y1, x2, y2 = coords
                draw.rectangle([x1, y1, x2, y2], outline=color, width=BBOX_WIDTH)

    return img


def wrap_text(text: str, max_chars: int = 80) -> str:
    """Simple word wrap."""
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > max_chars:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def make_panel(entry: dict) -> Image.Image:
    """Create a single panel showing all images + question + answer."""
    rel_paths = entry["image"]
    imgs = []
    for i, rel in enumerate(rel_paths):
        p = find_image(rel)
        if p is None:
            print(f"  WARNING: image not found: {rel}")
            imgs.append(Image.new("RGB", (320, 240), (80, 80, 80)))
        else:
            img = Image.open(p).convert("RGB")
            img = draw_annotations(img, entry, i)
            imgs.append(img)

    # Resize all images to same height
    target_h = 320
    resized = []
    for img in imgs:
        w, h = img.size
        new_w = int(w * target_h / h)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    # Concatenate horizontally
    total_w = sum(im.size[0] for im in resized) + 10 * (len(resized) - 1)
    img_strip = Image.new("RGB", (total_w, target_h), (30, 30, 30))
    x = 0
    for im in resized:
        img_strip.paste(im, (x, 0))
        x += im.size[0] + 10

    # Text area
    question = entry["conversations"][0]["value"]
    answer = entry["conversations"][1]["value"]
    qtype = entry["type"]
    entry_id = entry["id"]

    text = (
        f"Type: {qtype}  |  ID: {entry_id}\n\n"
        f"Q: {wrap_text(question, 100)}\n\n"
        f"A: {answer}"
    )

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    line_h = 22
    lines = text.split("\n")
    text_h = len(lines) * line_h + 30
    panel_w = max(total_w, 900)

    text_area = Image.new("RGB", (panel_w, text_h), (20, 20, 20))
    draw = ImageDraw.Draw(text_area)
    y = 10
    for line in lines:
        if line.startswith("Type:"):
            draw.text((10, y), line, fill=(255, 220, 60), font=font)
        elif line.startswith("Q:"):
            draw.text((10, y), line, fill=(180, 230, 255), font=font_small)
        elif line.startswith("A:"):
            draw.text((10, y), line, fill=(120, 255, 120), font=font)
        else:
            draw.text((10, y), line, fill=(200, 200, 200), font=font_small)
        y += line_h

    # Combine image strip + text
    canvas_w = max(img_strip.size[0], panel_w)
    canvas = Image.new("RGB", (canvas_w, target_h + text_h + 5), (30, 30, 30))
    canvas.paste(img_strip, (0, 0))
    canvas.paste(text_area, (0, target_h + 5))
    return canvas


def main():
    print(f"Loading {_TRAIN_JSON} ...")
    with open(_TRAIN_JSON) as f:
        data = json.load(f)

    # Collect first entry per type
    types_first: dict[str, dict] = {}
    for entry in data:
        t = entry["type"]
        if t not in types_first:
            types_first[t] = entry

    print(f"Found {len(types_first)} types:")
    for t in sorted(types_first):
        print(f"  {t}")

    _VIS_ROOT.mkdir(parents=True, exist_ok=True)

    for qtype, entry in sorted(types_first.items()):
        print(f"\nProcessing: {qtype} (id={entry['id']})")
        try:
            panel = make_panel(entry)
            out_path = _VIS_ROOT / f"{qtype}.jpg"
            panel.save(out_path, quality=92)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Results in {_VIS_ROOT}")


if __name__ == "__main__":
    main()
