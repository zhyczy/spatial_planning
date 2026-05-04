"""
probe_xyz_by_imgcount.py

Bucket xyz_validation results (atten_normal_*.json + atten_zero_*.json) by
the number of images per sample, and report Δaccuracy + win/loss within each
bucket. The "is xyz used at all" signal we already have (overall Δ); this
diagnostic asks "is xyz used the SAME WAY on single-image vs multi-image"?

Usage:
    python probe_xyz_by_imgcount.py \\
        --val_dir vis_results/xyz_val_atten_couple_spinbench/spinbench \\
        --jsonl   datasets/evaluation/spinbench_data/test.jsonl
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--jsonl",   required=True,
                    help="dataset test.jsonl with id + images fields")
    ap.add_argument("--id_key",  default="id")
    ap.add_argument("--img_key", default="images")
    args = ap.parse_args()

    # ── 1) load id -> n_images map from the dataset's jsonl ──────────────────
    id2n = {}
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            sid = row.get(args.id_key)
            imgs = row.get(args.img_key, [])
            if sid is not None:
                id2n[sid] = len(imgs)
    print(f"[probe] loaded {len(id2n)} samples with image-count from {args.jsonl}")

    # ── 2) load val results — normal + zero (keep all duplicates) ───────────
    def _load(prefix):
        rows = []
        for p in sorted(glob.glob(f"{args.val_dir}/atten_{prefix}_cuda*.json")):
            rows.extend(json.load(open(p)))
        return rows

    N_rows = _load("normal"); Z_rows = _load("zero")
    print(f"[probe] loaded normal={len(N_rows)}  zero={len(Z_rows)} rows "
          f"(can include duplicate ids when same image set has multiple questions)")

    # Pair by (id, question) — both passes process the same dataset list in
    # the same order, sharded across GPUs. Use position within each shard
    # OR fall back to (id, question) match.
    def _pair(N_rows, Z_rows):
        # Build (id, question) → row index for zero pass
        z_by_key = {}
        for r in Z_rows:
            k = (r.get("index"), r.get("question", "")[:200])
            z_by_key.setdefault(k, []).append(r)
        pairs = []
        for r in N_rows:
            k = (r.get("index"), r.get("question", "")[:200])
            cands = z_by_key.get(k)
            if cands:
                pairs.append((r, cands.pop(0)))
        return pairs

    pairs = _pair(N_rows, Z_rows)
    print(f"[probe] paired {len(pairs)} (id,question) tuples")

    # ── 3) bucket each PAIR by num_images ────────────────────────────────────
    buckets = defaultdict(list)
    fine    = defaultdict(list)
    n_unmapped = 0
    for n_row, z_row in pairs:
        sid = n_row.get("index")
        n_imgs = id2n.get(sid)
        if n_imgs is None:
            n_unmapped += 1
            continue
        bk = "single" if n_imgs == 1 else "multi"
        buckets[bk].append((n_row, z_row))
        fine[n_imgs].append((n_row, z_row))
    if n_unmapped:
        print(f"[probe] WARNING: {n_unmapped} pairs have no jsonl entry "
              f"(id mismatch).")

    def _stats(label, prs):
        if not prs: return
        n_corr = z_corr = 0
        n_diff = wins = losses = 0
        for n, z in prs:
            gt = n["answer"]
            nc, zc = (n["prediction"] == gt), (z["prediction"] == gt)
            n_corr += nc; z_corr += zc
            if n["prediction"] != z["prediction"]:
                n_diff += 1
                if nc and not zc: wins += 1
                elif zc and not nc: losses += 1
        n_total = len(prs)
        n_acc = n_corr / n_total * 100
        z_acc = z_corr / n_total * 100
        d_acc = z_acc - n_acc
        print(f"  {label:<14s}  n={n_total:4d}  "
              f"normal={n_acc:5.1f}%  zero={z_acc:5.1f}%  "
              f"Δ={d_acc:+6.2f}%  diff={n_diff:3d}  "
              f"win/loss={wins:3d}/{losses:3d}  "
              f"net={wins - losses:+4d}")

    print()
    print("=== bucketed by num_images (single = 1 image, multi = ≥2 images) ===")
    _stats("single", buckets["single"])
    _stats("multi",  buckets["multi"])
    _stats("ALL",    pairs)

    print()
    print("=== fine-grained by exact image count ===")
    for n_imgs in sorted(fine.keys()):
        _stats(f"n_imgs={n_imgs}", fine[n_imgs])


if __name__ == "__main__":
    main()
