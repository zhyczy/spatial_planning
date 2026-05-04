"""
probe_atten_bias.py

Diagnostic probe for the train_atten ckpt — does the per-layer
SpatialAttentionBias B actually carry useful per-pair signal, or is it
indistinguishable from noise (≪ attention-logit scale, near-equal to B(xyz=0))?

Per layer ℓ, on a real eval sample with N vision patches:
    B_real[ℓ] = MLP_ℓ(geom_features(xyz_real))    ∈ ℝ^{H × N × N}
    B_zero[ℓ] = MLP_ℓ(geom_features(zeros))       ∈ ℝ^{H × N × N}

We report:
    ‖B_real‖_F ,  ‖B_real-B_zero‖_F / ‖B_real‖_F ,  max|B_real| ,
    std(B_real) , corr( vec(B_real), vec(B_zero) )
plus an aggregate: mean over layers, % of layers where the relative
delta is < 0.1 (=B is dominated by the bias term, not the xyz signal).

For comparison: pre-softmax attention logits are ~O(1) after 1/√d scaling.
If max|B_real| ≪ 0.1 across layers, B contributes <10% to softmax inputs and
will be smoothed out — effectively noise.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.models import SpatialAttentionBias
from src.dataset import _qwen_align_view


def resize_xyz_block_mean(xyz: np.ndarray, mask: np.ndarray,
                          target_h: int, target_w: int) -> torch.Tensor:
    """Block-mean (stride_h, stride_w) → (target_h, target_w, 3).
    Mirrors src.dataset.train_dataset_qwen35.resize_xyz."""
    H, W = xyz.shape[:2]
    sh = H // target_h
    sw = W // target_w
    H_use = sh * target_h
    W_use = sw * target_w
    xyz = xyz[:H_use, :W_use].astype(np.float32)
    mask = mask[:H_use, :W_use].astype(bool)
    blocks = xyz.reshape(target_h, sh, target_w, sw, 3)
    valid = mask.reshape(target_h, sh, target_w, sw)
    cnt = valid.sum(axis=(1, 3))
    s = (blocks * valid[..., None]).sum(axis=(1, 3))
    mean = s / np.maximum(cnt, 1)[..., None]
    mean[cnt == 0] = 0.0
    return torch.from_numpy(mean.astype(np.float32))


def load_sample_xyz(sample_dir: Path, max_views: int = 4,
                    spatial_merge_size: int = 2,
                    qwen_factor: int = 28) -> torch.Tensor:
    """Load one sample's xyz, Qwen-align each view, block-mean down to
    LLM-patch grid, flatten across views → (N, 3)."""
    view_dirs = sorted(sample_dir.glob("view_*"))[:max_views]
    if not view_dirs:
        raise FileNotFoundError(f"no view_* under {sample_dir}")
    flat_chunks = []
    for vd in view_dirs:
        pts = np.load(vd / "pts3d.npy")           # (H, W, 3)
        mask_path = vd / "mask.npy"
        mask = np.load(mask_path).astype(bool) if mask_path.exists() \
               else np.ones(pts.shape[:2], dtype=bool)
        img_p = vd / "image.png"
        img = Image.open(img_p).convert("RGB")
        # Qwen-align all three together (same as training)
        img_q, pts_q, mask_q = _qwen_align_view(img, pts, mask)
        # Block-mean to LLM grid: H_q / qwen_factor patches × spatial_merge_size
        # so LLM grid stride = qwen_factor / spatial_merge_size = 14
        Hq, Wq = pts_q.shape[:2]
        llm_h = Hq // (qwen_factor // spatial_merge_size)
        llm_w = Wq // (qwen_factor // spatial_merge_size)
        # Actually patch_size=14, llm grid = Hq/(patch_size) // spatial_merge_size
        # = Hq / 14 / 2 = Hq / 28  (factor=28). Re-derive cleanly:
        llm_h = Hq // qwen_factor
        llm_w = Wq // qwen_factor
        xyz_grid = resize_xyz_block_mean(pts_q, mask_q, llm_h, llm_w)  # (llm_h, llm_w, 3)
        flat_chunks.append(xyz_grid.reshape(-1, 3))
    return torch.cat(flat_chunks, dim=0)


def build_bias_modules_from_ckpt(ckpt_dir: Path,
                                 num_heads: int,
                                 hidden_dim: int = 128,
                                 device: str = "cuda:0",
                                 dtype=torch.bfloat16,
                                 ) -> dict[str, SpatialAttentionBias]:
    """Load spatial_bias.pt and instantiate one SpatialAttentionBias per layer."""
    state = torch.load(str(ckpt_dir / "spatial_bias.pt"), map_location="cpu")
    layers: dict[str, SpatialAttentionBias] = {}
    for full_name, sub_state in state.items():
        # Heuristic: extract layer index from key like
        # "...language_model.layers.<idx>.self_attn..." or similar.
        idx_token = "language_model.layers."
        i = full_name.find(idx_token)
        if i >= 0:
            tail = full_name[i + len(idx_token):]
            layer_idx = tail.split(".")[0]
            tag = f"L{int(layer_idx):02d}"
        else:
            tag = full_name.replace("/", "_").replace(".", "_")
        m = SpatialAttentionBias(num_heads=num_heads,
                                 hidden_dim=hidden_dim, zero_init=False)
        m.load_state_dict(sub_state)
        m.to(device=device, dtype=dtype)
        m.eval()
        layers[tag] = m
    return layers


@torch.no_grad()
def per_layer_stats(bias_mods: dict[str, SpatialAttentionBias],
                    xyz: torch.Tensor,
                    device: str = "cuda:0",
                    dtype=torch.bfloat16) -> dict[str, dict]:
    """For each layer, return stats of B_real and ΔB = B_real - B_zero."""
    N = xyz.shape[0]
    xyz_real = xyz.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,N,3)
    xyz_zero = torch.zeros_like(xyz_real)
    vmask = torch.ones(1, N, dtype=torch.bool, device=device)

    out = {}
    for tag, mod in bias_mods.items():
        B_real = mod(xyz_real, vmask).float()    # (1, H, N, N)
        B_zero = mod(xyz_zero, vmask).float()
        delta  = B_real - B_zero
        # restrict to V×V block (it IS the full thing here since seq_len=N)
        # vector-form for correlation
        vR = B_real.reshape(-1)
        vZ = B_zero.reshape(-1)
        if vR.std() > 0 and vZ.std() > 0:
            corr = ((vR - vR.mean()) * (vZ - vZ.mean())).mean() / (vR.std() * vZ.std())
            corr = float(corr)
        else:
            corr = float("nan")
        out[tag] = dict(
            n=N,
            mean_real=float(B_real.mean()),
            std_real =float(B_real.std()),
            absmax_real=float(B_real.abs().max()),
            fro_real =float(B_real.norm()),
            fro_zero =float(B_zero.norm()),
            fro_delta=float(delta.norm()),
            rel_delta=float(delta.norm() / (B_real.norm() + 1e-12)),
            corr_real_zero=corr,
        )
    return out


def aggregate(per_sample: list[dict[str, dict]]) -> dict:
    layers = list(per_sample[0].keys())
    agg = {}
    for L in layers:
        rec = {}
        for k in per_sample[0][L]:
            vals = np.array([s[L][k] for s in per_sample], dtype=np.float64)
            rec[k] = float(np.nanmean(vals))
        agg[L] = rec
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="train_records/atten_*/step_* dir with spatial_bias.pt")
    ap.add_argument("--data_dir", default="datasets/evaluation/MindCube",
                    help="dataset root with 3d_results/<id>/view_*")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_views", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=32,
                    help="Qwen3.5-4B has 32 KV heads (default).")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="Optional JSON output path for per-layer stats.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt).resolve()
    data_dir = Path(args.data_dir).resolve()

    # ── 1) load the per-layer bias modules from the checkpoint ──────────────
    print(f"[probe] loading {ckpt_dir/'spatial_bias.pt'} ...", flush=True)
    bias_mods = build_bias_modules_from_ckpt(
        ckpt_dir, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"[probe] loaded {len(bias_mods)} SpatialAttentionBias modules "
          f"(num_heads={args.num_heads}, hidden_dim={args.hidden_dim})",
          flush=True)

    # Quick MLP-weight summary
    w2_norms = []
    w1_norms = []
    for tag, m in bias_mods.items():
        w2_norms.append(float(m.mlp[-1].weight.norm()))
        w1_norms.append(float(m.mlp[0].weight.norm()))
    w1_arr = np.array(w1_norms); w2_arr = np.array(w2_norms)
    print(f"[probe] ‖W₁‖ across layers: mean={w1_arr.mean():.3f} "
          f"min={w1_arr.min():.3f} max={w1_arr.max():.3f}", flush=True)
    print(f"[probe] ‖W₂‖ across layers: mean={w2_arr.mean():.3f} "
          f"min={w2_arr.min():.3f} max={w2_arr.max():.3f}", flush=True)
    print(f"[probe] (W₂ trained from zero-init; tiny ‖W₂‖ ⇒ B≈0 ⇒ xyz pathway "
          f"is effectively a noise channel.)", flush=True)

    # ── 2) load a few real samples ──────────────────────────────────────────
    res_root = data_dir / "3d_results"
    sample_dirs = sorted([p for p in res_root.iterdir()
                          if p.is_dir()])[:args.n_samples]
    if not sample_dirs:
        raise FileNotFoundError(f"no samples under {res_root}")
    print(f"[probe] using {len(sample_dirs)} samples from {res_root}", flush=True)

    per_sample_stats = []
    for sd in sample_dirs:
        try:
            xyz = load_sample_xyz(sd, max_views=args.max_views)
        except Exception as e:
            print(f"[probe]   skip {sd.name}: {e}")
            continue
        N = xyz.shape[0]
        stats = per_layer_stats(bias_mods, xyz, device=args.device)
        per_sample_stats.append(stats)
        # Per-sample summary
        absmax = np.array([s["absmax_real"] for s in stats.values()])
        rel_d  = np.array([s["rel_delta"]   for s in stats.values()])
        corr   = np.array([s["corr_real_zero"] for s in stats.values()])
        print(f"[probe] sample={sd.name[:40]:40s}  N={N:4d}  "
              f"max|B|≈{absmax.mean():.4f}  rel_Δ≈{rel_d.mean():.3f}  "
              f"corr(B_real,B_zero)≈{corr.mean():+.3f}",
              flush=True)

    # ── 3) aggregate across samples ─────────────────────────────────────────
    if not per_sample_stats:
        raise RuntimeError("no successful samples")
    agg = aggregate(per_sample_stats)

    # Per-layer table
    print()
    print(f"{'layer':>6s} | {'‖B‖_F':>9s} {'max|B|':>9s} {'std(B)':>9s} | "
          f"{'‖ΔB‖/‖B‖':>10s} {'corr':>7s}")
    print("-" * 64)
    for tag in sorted(agg.keys()):
        s = agg[tag]
        print(f"{tag:>6s} | {s['fro_real']:>9.4f} {s['absmax_real']:>9.4f} "
              f"{s['std_real']:>9.5f} | {s['rel_delta']:>10.3f} "
              f"{s['corr_real_zero']:>+7.3f}")

    # Aggregate summary
    arr = lambda k: np.array([agg[L][k] for L in agg])
    print()
    print(f"[probe] aggregate over {len(agg)} layers:")
    print(f"          mean ‖B_real‖_F     = {arr('fro_real').mean():.4f}")
    print(f"          mean max|B_real|    = {arr('absmax_real').mean():.4f}  "
          f"(softmax-input scale ~ O(1); <0.1 = negligible)")
    print(f"          mean ‖ΔB‖/‖B_real‖  = {arr('rel_delta').mean():.3f}  "
          f"(≪1 ⇒ B≈B(xyz=0); xyz contributes little)")
    print(f"          mean corr(real,0)   = {arr('corr_real_zero').mean():+.3f}  "
          f"(near +1 ⇒ same map regardless of xyz)")
    n_low = int((arr("absmax_real") < 0.1).sum())
    n_dom = int((arr("rel_delta") < 0.1).sum())
    print(f"          layers with max|B|<0.1 : {n_low}/{len(agg)}")
    print(f"          layers with rel_Δ<0.1  : {n_dom}/{len(agg)}  "
          "(xyz signal smaller than constant offset)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_layer": agg,
                       "summary": {
                           "n_layers": len(agg),
                           "mean_fro_real": float(arr("fro_real").mean()),
                           "mean_absmax_real": float(arr("absmax_real").mean()),
                           "mean_rel_delta": float(arr("rel_delta").mean()),
                           "mean_corr_real_zero": float(arr("corr_real_zero").mean()),
                           "n_layers_absmax_lt_0.1": n_low,
                           "n_layers_rel_delta_lt_0.1": n_dom,
                       }}, f, indent=2)
        print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
