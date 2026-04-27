"""
Spatial Attention Bias — per-layer learnable geometric bias for vision-vision attention.

Step 1 of the spatial-attention-block work: standalone module, NOT yet wired
into the transformer's attention. Tested independently first.

────────────────────────────────────────────────────────────────────────────────
Design
────────────────────────────────────────────────────────────────────────────────
For each LLM transformer layer, augment standard attention with an additive
geometric bias B ∈ ℝ^{H × L × L} that is non-zero only on vision-vision pairs:

    attention = softmax( QK^T / √d  +  causal_mask  +  B ) V

For the N vision tokens with 3-D scene coordinates p_1, ..., p_N ∈ ℝ³ ,
think of geometry as 4 stacked N×N "channels":

    [ M_nx | M_ny | M_nz | M_d ]   ∈ ℝ^{N × N × 4}                 (n_x, n_y, n_z, d  per pair)

then pass each (i,j) entry through a 2-layer MLP that fuses the 4 channels
into H per-head bias scalars:

    Δp_ij  = p_j - p_i  ∈ ℝ³                  (signed component diff)
    d_ij   = ‖Δp_ij‖₂                          (≥ 0, scalar)
    n_ij   = Δp_ij / d_ij                      (unit direction; scale-decoupled)
    feat_ij = (n_x, n_y, n_z, d) ∈ ℝ⁴
    B[h, i, j] = ( W₂ · GELU( W₁ · feat_ij ) )[h]    ← 2-layer MLP, per-head, per-layer

Why decouple direction from magnitude: a single Linear over (Δx, Δy, Δz, d) entangles
orientation with absolute scale (a vector twice as long produces twice the activation
on its direction channels). Normalizing direction lets W₁ specialize on geometry while
the distance channel carries scale separately. The 2-layer MLP with GELU adds the
non-linearity needed to combine direction and distance into per-head biases.

Why per-head: different attention heads tend to specialize on different
geometric patterns (e.g. vertical neighbors vs. far-distance pairs); a per-head
output gives each head its own learnable spatial template, while the per-pair
4-D input is shared.

Block structure of B over the full sequence (vision-mask drives the scatter,
visualized assuming vision-first ordering for clarity):

            ┌─────────────────────────┬───────────────────┐
            │ B_spatial (V → V)       │     0  (V → T)    │
            ├─────────────────────────┼───────────────────┤
            │     0   (T → V)         │     0  (T → T)    │
            └─────────────────────────┴───────────────────┘

Decode-phase note (handled by the wrapper, not here): when the current query is
a single newly-generated text token, its row of B is all zero — vision-aware
reasoning lives in the KV cache built during prefill. The wrapper can simply
short-circuit `bias=None` during decode.

Init scheme: only the OUTPUT layer (W₂) is zero-initialized → B = 0 at init →
pretrained behavior is preserved on step 0. W₁ uses default Kaiming-uniform so
gradients flow normally from the start.
"""

import torch
import torch.nn as nn


class SpatialAttentionBias(nn.Module):
    """
    Per-layer learnable bias
        B[h,i,j] = MLP_h(n_x, n_y, n_z, ‖Δp‖)_ij
    where (n_x,n_y,n_z) is the unit direction of (p_j − p_i) and ‖Δp‖ is its
    magnitude. Applied only on vision-vision attention pairs.

    Args:
        num_heads:  H — number of attention heads in the host attention layer
                    (the MLP outputs one scalar per head per pair).
        hidden_dim: width of the MLP hidden layer. Default 128 — gives each
                    head enough independent non-linear basis functions over
                    the 4-D edge feature without inflating params (~170K total
                    on a 36-layer / 32-head model, ~1% of a rank-16 LoRA).
        zero_init:  if True (default), the OUTPUT layer (W₂) inits to zero so
                    that B == 0 at step 0 (pretrained model undisturbed).
    """

    def __init__(
        self,
        num_heads:  int,
        hidden_dim: int  = 128,
        zero_init:  bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        # ── Two-layer MLP, applied INDEPENDENTLY (with shared weights) at
        #    every (i, j) pair location. Equivalent to a 1×1 conv over the
        #    N×N "edge image" with 4 input channels and `num_heads` output
        #    channels. Same MLP is reused for all N² pairs and all batches.
        #
        #  Conceptual data flow (suppose B=1, N_vis=N, num_heads=H, hidden=h):
        #
        #     edge feature (input)        : (1,  N,  N,  4)
        #              │                     ↑   ↑   ↑   ↑
        #              │              batch  i   j   ch.   ← 4 = (n_x, n_y, n_z, d)
        #              │
        #              ▼   self.mlp[0] = nn.Linear(4 → h)
        #                       weight: (h, 4)     bias: (h,)
        #                       params: 4·h + h
        #                       (broadcast over the leading B,N,N dims)
        #
        #     after Linear-1               : (1,  N,  N,  h)
        #              │
        #              ▼   self.mlp[1] = GELU (point-wise, no shape change)
        #
        #     after GELU                   : (1,  N,  N,  h)
        #              │
        #              ▼   self.mlp[2] = nn.Linear(h → H)
        #                       weight: (H, h)     bias: (H,)
        #                       params: h·H + H
        #
        #     output (per-head bias logits): (1,  N,  N,  H)
        #              │
        #              ▼  permute(0, 3, 1, 2) in forward()
        #
        #     final M_vis                  : (1,  H,  N,  N)
        #              ↑   ↑    ↑   ↑
        #          batch head row col          ← ready to scatter into bias
        #
        #  Per-layer params  : (4·h + h) + (h·H + H) = (4+H)·h + (1+H)
        #     h=128, H=32  →  4·128 + 128 + 128·32 + 32 = 512+128+4096+32 = 4768
        #     × 36 LLM layers  →  ~172K total trainable params.
        #
        #  Per-layer FLOPs (per forward, per pair): 4·h + h·H multiplies.
        #     With N≈256 (4 imgs × 64 patches) → N² = 65K pairs.
        #     65K · (4·128 + 128·32) ≈ 0.3G MAC per layer × 36 ≈ 11G total.
        #
        #  Kept as nn.Sequential so state_dict keys are mlp.0.weight /
        #  mlp.0.bias / mlp.2.weight / mlp.2.bias — clean for ckpt save/load.
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),         # W₁: (hidden_dim, 4),  b₁: (hidden_dim,)
            nn.GELU(),                        # point-wise
            nn.Linear(hidden_dim, num_heads), # W₂: (num_heads, hidden_dim),  b₂: (num_heads,)
        )

        if zero_init:
            # Only zero the OUTPUT layer (W₂, b₂). That alone forces M_vis ≡ 0
            # at step 0 → B ≡ 0 → pretrained attention is undisturbed.
            # W₁ keeps PyTorch's default Kaiming-uniform init so the gradient
            # path through Linear-1 is healthy from step 1 onward.
            #   (If we also zero W₁, then ∂L/∂W₂ = h(x)ᵀ · grad ≡ 0 at step 0
            #    because h(x) = GELU(W₁x + b₁) = GELU(0) ≠ 0 actually — but
            #    the gradient flowing INTO W₁ is W₂ᵀ · grad which is zero,
            #    so W₁ can't move. Keeping W₁ at Kaiming avoids this trap.)
            nn.init.zeros_(self.mlp[-1].weight)   # (num_heads, hidden_dim) → all zeros
            nn.init.zeros_(self.mlp[-1].bias)     # (num_heads,)            → all zeros

    @staticmethod
    def compute_geometry_features(xyz: torch.Tensor) -> torch.Tensor:
        """
        Build the (n_x, n_y, n_z, ‖Δp‖) edge feature for N points — direction is
        unit-normalized (scale-decoupled), distance is kept as a raw scalar.

        Args:
            xyz: (..., N, 3) float
        Returns:
            feat: (..., N, N, 4) where
                feat[..., i, j, 0:3] = (xyz[..., j, :] - xyz[..., i, :]) / d_ij
                feat[..., i, j, 3]   = d_ij = ‖xyz[..., j, :] - xyz[..., i, :]‖₂
        """
        # diff[..., i, j, :] = p_j - p_i  (Query i → Key j vector).
        # unsqueeze(-3) broadcasts j over rows; unsqueeze(-2) broadcasts i over cols.
        diff = xyz.unsqueeze(-3) - xyz.unsqueeze(-2)        # (..., N, N, 3)
        # +eps inside sqrt: at i==j diff is zero, and ‖·‖ has undefined grad at 0
        # → NaN under autograd / gradient checkpointing recompute.
        dist = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)
        # Scale-decouple: unit direction. Same eps in the divisor protects the
        # i==j row/col where diff≈0 (resulting unit vector is ≈0, harmless).
        direction = diff / dist
        return torch.cat([direction, dist], dim=-1)         # (..., N, N, 4)

    def forward(
        self,
        xyz: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute B for one layer over a (possibly padded) batched sequence.

        Args:
            xyz:         (B, N_max, 3) per-vision-token 3-D coords. Upstream
                         (SpatialAttnVanillaModel.forward) has flattened each
                         sample's per-image xyz tensors into a single (N_b, 3)
                         tensor and then `pad_sequence`-d them to a common
                         N_max across the batch. So:
                             • B is the real batch size (>= 1)
                             • Per sample b, the first N_b entries are valid
                               world-frame coords; positions [N_b : N_max] are
                               zero-padding.
                             • Coordinates within one sample share the world
                               frame (= view_0000 camera frame for MindCube),
                               so the N_b × N_b sub-block contains BOTH intra-
                               and inter-image patch pairs.
                         For each sample we recover N_b at runtime via
                         vision_mask[b].sum(); padded entries are sliced off.
            vision_mask: (B, seq_len) bool — True at vision-token positions.
                         vision_mask[b].sum() must equal that sample's N_b
                         (asserted upstream in SpatialAttnVanillaModel.forward).

        Returns:
            bias: (B, H, seq_len, seq_len) — same dtype/device as the proj
                  weights. Non-zero only on entries where both row and column
                  are vision tokens; everywhere else exactly 0.
        """
        B, N_max, _ = xyz.shape          # B is the *batch* dim (was hardcoded
                                         # to 1; now genuinely variable). Within
                                         # a sample, multi-image patches are
                                         # already flattened upstream.
        seq_len = vision_mask.shape[1]
        H = self.num_heads
        proj_dtype = self.mlp[0].weight.dtype

        # 1) Pairwise edge features (n_x, n_y, n_z, d) on the FULL padded
        #    grid — (B, N_max, N_max, 4). Padded rows/cols compute non-trivial
        #    features but are sliced out before scatter (step 3), so they
        #    cost FLOPs but contribute zero gradient (no path to loss).
        feat = self.compute_geometry_features(xyz)

        # 2) Two-layer MLP → per-head bias  (B, N_max, N_max, H) → (B, H, N_max, N_max)
        M_all = self.mlp(feat.to(proj_dtype))
        M_all = M_all.permute(0, 3, 1, 2).contiguous()

        # 3) Scatter the *valid* N_b × N_b sub-block of each sample into the
        #    full attention grid; everything else stays zero.
        #    The for-loop is over the batch axis. For each sample we:
        #      • read its true vision-token count N_b from vision_mask
        #      • pull the top-left N_b × N_b sub-block from M_all (the rest
        #        is computed from padded zeros and discarded)
        #      • scatter it into bias at the (vis_idx × vis_idx) cross-section
        bias = torch.zeros(
            B, H, seq_len, seq_len,
            dtype=M_all.dtype, device=M_all.device,
        )
        for b in range(B):
            vis_idx = vision_mask[b].nonzero(as_tuple=True)[0]   # (N_b,)
            N_b = vis_idx.numel()
            if N_b == 0:
                continue
            # bias[b, :, vis_idx[i], vis_idx[j]] = M_all[b, :, i, j]  for i,j < N_b
            bias[b, :, vis_idx[:, None], vis_idx[None, :]] = M_all[b, :, :N_b, :N_b]

        return bias
