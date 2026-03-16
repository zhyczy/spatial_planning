"""
Demo: run CoordEstimator on two MindCube samples.

Usage
-----
    conda run -n spi python spatial_planning/demo_coord_esti.py
or, from inside spatial_planning/:
    python demo_coord_esti.py
"""

import json
import sys
from pathlib import Path

# Make sure spatial_planning/ is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coord_esti import CoordEstimator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MINDCUBE_DIR = _HERE / "datasets" / "evaluation" / "MindCube"
JSONL_FILE   = MINDCUBE_DIR / "MindCube_tinybench.jsonl"

# ---------------------------------------------------------------------------
# Load two samples from MindCube
# ---------------------------------------------------------------------------
samples = []
with open(JSONL_FILE) as f:
    for line in f:
        samples.append(json.loads(line.strip()))
        if len(samples) == 2:
            break

print("=" * 60)
print("MindCube demo — 2 samples")
print("=" * 60)

# ---------------------------------------------------------------------------
# Build absolute image path lists
# ---------------------------------------------------------------------------
def abs_paths(sample):
    return [str(MINDCUBE_DIR / p) for p in sample["images"]]

sample_image_lists = [abs_paths(s) for s in samples]

for i, (s, paths) in enumerate(zip(samples, sample_image_lists)):
    print(f"\nSample {i}  id={s['id']}")
    print(f"  question : {s['question'][:80]}…")
    print(f"  images   : {[Path(p).name for p in paths]}")

# ---------------------------------------------------------------------------
# Run estimation (model loaded once, reused for both samples)
# ---------------------------------------------------------------------------
estimator = CoordEstimator()

for i, (sample, paths) in enumerate(zip(samples, sample_image_lists)):
    run_name = f"mindcube_{sample['id']}"
    print(f"\n[Demo {i}] Running inference on sample '{sample['id']}' …")

    results = estimator.estimate(paths, run_name=run_name)

    print(f"[Demo {i}] Results ({len(results)} views):")
    for v, r in enumerate(results):
        print(
            f"  view {v}: pts3d {r['pts3d'].shape}  "
            f"depth {r['depth'].shape}  "
            f"pose {r['camera_pose'].shape}  "
            f"K {r['intrinsics'].shape}  "
            f"valid_px={r['mask'].sum()}"
        )
        print(f"    intrinsics (fx,fy,cx,cy): "
              f"{r['intrinsics'][0,0]:.1f}, {r['intrinsics'][1,1]:.1f}, "
              f"{r['intrinsics'][0,2]:.1f}, {r['intrinsics'][1,2]:.1f}")

print("\nDone. Results saved under spatial_planning/coord_results/")
