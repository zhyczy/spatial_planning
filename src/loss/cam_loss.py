import torch
from itertools import combinations


def geodesic_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean rotation-angle loss between predicted and GT rotation matrices.
    R_pred, R_gt: (N, 3, 3)
    """
    # Compute in float32: acos gradient blows up near ±1 in bfloat16
    R_pred = R_pred.float()
    R_gt   = R_gt.float()
    R_diff = torch.bmm(R_pred, R_gt.transpose(-1, -2))         # (N, 3, 3)
    trace  = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)         # (N,)
    cos    = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()


def cycle_consistency_loss(
    R_preds: torch.Tensor,
    t_preds: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotation + translation cycle-consistency loss.

    Rotation (Frobenius norm):
      Pair round-trip (i < j):   ||R_{i→j} @ R_{j→i} - I||_F
      Triangle {a, b, c}:        ||R_{a→b} @ R_{b→c} @ R_{c→a} - I||_F

    Translation (L1 norm, more robust to outliers):
      Pair round-trip (i < j):   ||R_{i→j} @ t_{j→i} + t_{i→j}||_1
      Triangle {a, b, c}:        ||R_{a→b} @ R_{b→c} @ t_{c→a}
                                    + R_{a→b} @ t_{b→c} + t_{a→b}||_1

    Args:
      R_preds: (K, 3, 3)  where K = N*(N-1), ordered by
               [(i,j) for i in range(N) for j in range(N) if i != j]
      t_preds: (K, 3)     same ordering as R_preds
      N:       number of views

    Returns:
      (cycle_r, cycle_t) — two scalars, mean over all terms.
      Returns (0, 0) when N < 2.
    """
    zero = R_preds.new_tensor(0.0)
    if N < 2:
        return zero, zero

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    p2k   = {p: k for k, p in enumerate(pairs)}
    I3    = torch.eye(3, device=R_preds.device, dtype=R_preds.dtype)

    r_errs = []
    t_errs = []

    # Pair round-trips for all i < j
    for i in range(N):
        for j in range(i + 1, N):
            kij, kji = p2k[(i, j)], p2k[(j, i)]
            # Rotation: R_{i→j} @ R_{j→i} = I
            R_cycle = R_preds[kij] @ R_preds[kji]
            r_errs.append(torch.norm(R_cycle - I3, p="fro"))
            # Translation: R_{i→j} @ t_{j→i} + t_{i→j} = 0
            t_cycle = R_preds[kij] @ t_preds[kji] + t_preds[kij]
            t_errs.append(t_cycle.abs().sum())          # L1 norm

    # Triangle cycles for all triples {a, b, c}
    for a, b, c in combinations(range(N), 3):
        kab, kbc, kca = p2k[(a, b)], p2k[(b, c)], p2k[(c, a)]
        # Rotation: R_{a→b} @ R_{b→c} @ R_{c→a} = I
        R_cycle = R_preds[kab] @ R_preds[kbc] @ R_preds[kca]
        r_errs.append(torch.norm(R_cycle - I3, p="fro"))
        # Translation: R_{a→b} @ R_{b→c} @ t_{c→a} + R_{a→b} @ t_{b→c} + t_{a→b} = 0
        t_cycle = (R_preds[kab] @ R_preds[kbc] @ t_preds[kca]
                   + R_preds[kab] @ t_preds[kbc]
                   + t_preds[kab])
        t_errs.append(t_cycle.abs().sum())              # L1 norm

    cycle_r = torch.stack(r_errs).mean()
    cycle_t = torch.stack(t_errs).mean()
    return cycle_r, cycle_t