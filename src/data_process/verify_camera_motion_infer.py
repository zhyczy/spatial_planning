"""
verify_camera_motion_infer.py

Verify that camera_motion_infer GT answers in train_10k.json are consistent
with the camera poses stored in reconstruct/{id}.npz.

For each sample:
  1. Load poses_ff from reconstruct npz (cam-to-first-frame, metres)
  2. cam1's centre in cam0's frame = poses_ff[1][:3, 3]  (since poses_ff[0] = I)
  3. Project into cam0's image plane using K_color
     x = K[0,0]*X/Z + K[0,2],  y = K[1,1]*Y/Z + K[1,2],  depth = Z
  4. Parse GT answer (3 formats: MC select / compact fill / natural-language fill)
  5. Report pixel error and depth error vs GT

Answer formats handled:
  MC select   : options "A. Image Coor:(X, Y), Depth:D meters" → answer is letter
  Compact fill: "(X,Y),D"
  NL fill     : "...positioned at (X,Y), with a depth of D meters."

Usage
-----
    cd spatial_planning
    python src/data_process/verify_camera_motion_infer.py --n 10
    python src/data_process/verify_camera_motion_infer.py --n 10 --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from reconstruct_3d import (
    load_intrinsics,
    _scene_dir,
    _scene_id_from_path,
    _frame_id_from_path,
)

_SPAR_ROOT       = Path(__file__).resolve().parents[2] / "datasets/train/SPAR_7M/spar"
_RECONSTRUCT_DIR = _SPAR_ROOT / "reconstruct"
_TRAIN_JSON      = _SPAR_ROOT / "train_10k.json"


# ---------------------------------------------------------------------------
# Answer parsers
# ---------------------------------------------------------------------------

# MC option line: "A. Image Coor:(932, 468), Depth:1.9 meters"
_MC_OPT_RE = re.compile(
    r"([A-D])\.\s*Image Coor:\((\d+),\s*(\d+)\),\s*Depth:([\d.]+)\s*meters?",
    re.IGNORECASE,
)
# GT answer for compact fill: "(606,69),2.8"
_FILL_COMPACT_RE = re.compile(r"\((\d+),\s*(\d+)\)[,\s]+([\d.]+)")
# GT answer for NL fill: "at (648,281), with a depth of 3.8 meters"
_FILL_NL_RE = re.compile(r"at \((\d+),\s*(\d+)\).*depth of ([\d.]+)")


def parse_mc_options(question: str):
    """Return {letter: (x, y, depth)} if MC format, else {}."""
    return {
        m.group(1): (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        for m in _MC_OPT_RE.finditer(question)
    }


def parse_fill_answer(answer: str):
    """Parse fill-format GT answer → (x, y, depth) or None."""
    m = _FILL_COMPACT_RE.search(answer)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    m = _FILL_NL_RE.search(answer)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return None


# ---------------------------------------------------------------------------
# Compute projected (x, y, depth) of cam1 centre in cam0 image plane
# ---------------------------------------------------------------------------

# GT answers were generated at target_W=1000px wide images.
# Scale K accordingly: s = target_W / orig_W  where orig_W ≈ 2*cx.
_GT_TARGET_W = 1000.0


def compute_projection(entry: dict, rec_dir: Path, spar_root: Path):
    """Returns (x_proj, y_proj, depth_m, K_scaled) or raises on error.

    Scales K to match the ~1000px-wide images used for GT generation.
    """
    npz_path = rec_dir / f"{entry['id']}.npz"
    data = np.load(npz_path)
    poses   = data["poses"]          # (N, 4, 4) cam-to-first-frame
    dataset = str(data["dataset"])

    # cam1 centre in cam0 frame = translation of poses[1]
    # (poses[0] = identity, so first-frame == cam0 frame)
    cam1_centre = poses[1, :3, 3]    # (3,) in metres

    # Load cam0 colour intrinsics
    img_rel      = entry["image"][0]
    scene_id_img = _scene_id_from_path(img_rel)
    frame_id     = _frame_id_from_path(img_rel)
    sd           = _scene_dir(str(spar_root), dataset, scene_id_img)
    _, K_color   = load_intrinsics(dataset, sd, frame_id)  # (3, 3)

    # GT coordinates are in 1000×1000 square-resized image space.
    # Apply independent X and Y scales:  scale_x = 1000/(2*cx),  scale_y = 1000/(2*cy)
    cx, cy   = K_color[0, 2], K_color[1, 2]
    scale_x  = _GT_TARGET_W / (cx * 2)   # 1000 / orig_W
    scale_y  = _GT_TARGET_W / (cy * 2)   # 1000 / orig_H

    X, Y, Z = cam1_centre

    if Z == 0:
        raise ValueError(f"Camera 1 exactly at camera 0 image plane (Z=0)")

    x_proj = scale_x * (K_color[0, 0] * X / Z + cx)
    y_proj = scale_y * (K_color[1, 1] * Y / Z + cy)
    return x_proj, y_proj, Z, K_color


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify(n: int, verbose: bool):
    with open(_TRAIN_JSON) as f:
        data = json.load(f)

    samples = [d for d in data if d.get("type") == "camera_motion_infer"][:n]
    print(f"Verifying {len(samples)} camera_motion_infer samples\n")

    match = mismatch = skip = 0

    for s in samples:
        eid   = s["id"]
        q     = s["conversations"][0]["value"]
        gt_raw = s["conversations"][1]["value"].strip()

        # --- detect format ---
        mc_opts = parse_mc_options(q)
        is_mc   = bool(mc_opts)

        # --- compute ground-truth (x, y, depth) ---
        if is_mc:
            gt_letter = gt_raw.upper()
            gt_xyz = mc_opts.get(gt_letter)
            fmt = "MC"
        else:
            gt_xyz = parse_fill_answer(gt_raw)
            fmt = "fill"

        if gt_xyz is None:
            if verbose:
                print(f"[{eid}] SKIP — unparseable answer: {repr(gt_raw[:80])}")
            skip += 1
            continue

        # --- compute from camera pose ---
        try:
            x_p, y_p, depth, K = compute_projection(s, _RECONSTRUCT_DIR, _SPAR_ROOT)
        except Exception as e:
            if verbose:
                print(f"[{eid}] ERROR — {e}")
            skip += 1
            continue

        gt_x, gt_y, gt_d = gt_xyz
        pix_err = ((gt_x - x_p) ** 2 + (gt_y - y_p) ** 2) ** 0.5
        dep_err = abs(gt_d - depth)

        # For MC: find closest option and check it equals GT
        if is_mc:
            def _score(opt):
                ox, oy, od = opt
                return ((ox - x_p)**2 + (oy - y_p)**2)**0.5 + abs(od - depth) * K[0,0]
            best_letter = min(mc_opts, key=lambda l: _score(mc_opts[l]))
            ok = (best_letter == gt_letter)
        else:
            # Fill: match if pixel error < 30px and depth error < 0.3m
            ok = pix_err < 30 and dep_err < 0.3

        if ok:
            match += 1
        else:
            mismatch += 1

        if verbose or not ok:
            status = "✓" if ok else "✗"
            print(f"[{eid}] {status} [{fmt}]")
            print(f"  Computed : x={x_p:.1f}  y={y_p:.1f}  depth={depth:.3f}m")
            print(f"  GT       : x={gt_x}  y={gt_y}  depth={gt_d}m")
            print(f"  Error    : pix={pix_err:.1f}px  depth={dep_err:.3f}m")
            if is_mc and not ok:
                best_xyz = mc_opts[best_letter]
                print(f"  Best opt : {best_letter} → {best_xyz}  (GT={gt_letter})")
            print()

    total_valid = match + mismatch
    print("=" * 55)
    print(f"match={match}  mismatch={mismatch}  skip/error={skip}  total={n}")
    if total_valid > 0:
        print(f"Accuracy: {match}/{total_valid} = {match/total_valid*100:.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",       type=int, default=10, help="Number of samples to verify")
    p.add_argument("--verbose", action="store_true",  help="Print all samples, not just failures")
    args = p.parse_args()
    verify(args.n, args.verbose)


if __name__ == "__main__":
    main()
