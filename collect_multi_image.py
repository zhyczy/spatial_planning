"""
Collect all multi-image (non-video) QA entries from SPAR_7M/spar subfolders.
Skips: single-image entries, video entries, and question types:
  nav, appearance_order, room_size, obj_frame_locate
Outputs: train.json and val.json in the spar root directory.
"""

import json
import os
from pathlib import Path

SPAR_ROOT = Path("/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/train/SPAR_7M/spar")

SKIP_TYPES = {"nav", "appearance_order", "room_size", "obj_frame_locate"}

SUBFOLDERS = ["scannet", "scannetpp", "structured3d"]


def is_video_entry(entry, question_type_dir):
    """Return True if entry involves video frames."""
    # Check folder name
    if "video" in question_type_dir.lower():
        return True
    # Check image paths for video_color pattern
    images = entry.get("image", [])
    if any("video_color" in str(img) for img in images):
        return True
    return False


def collect_split(split):
    """Collect all multi-image entries for a given split (train or val)."""
    all_entries = []
    stats = {}

    for subfolder in SUBFOLDERS:
        split_dir = SPAR_ROOT / subfolder / "qa_jsonl" / split
        if not split_dir.exists():
            print(f"  [skip] {subfolder}/{split} — directory not found")
            continue

        subfolder_count = 0
        for qtype_dir in sorted(split_dir.iterdir()):
            if not qtype_dir.is_dir():
                continue

            qtype_name = qtype_dir.name

            # Skip unwanted question types
            if qtype_name in SKIP_TYPES:
                continue

            # Skip video question types (folder name contains "video")
            if "video" in qtype_name.lower():
                continue

            # Process all JSONL files in this question type directory (recursively)
            for jsonl_file in sorted(qtype_dir.rglob("*.jsonl")):
                with open(jsonl_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        images = entry.get("image", [])

                        # Skip single-image entries
                        if len(images) <= 1:
                            continue

                        # Skip video entries (check image paths)
                        if is_video_entry(entry, qtype_name):
                            continue

                        all_entries.append(entry)
                        subfolder_count += 1

        stats[subfolder] = subfolder_count
        print(f"  {subfolder}/{split}: {subfolder_count} entries")

    return all_entries, stats


def main():
    for split in ["train", "val"]:
        print(f"\nCollecting {split} split...")
        entries, stats = collect_split(split)

        out_path = SPAR_ROOT / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(entries, f)

        print(f"  Total {split}: {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
