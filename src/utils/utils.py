# ── rotation utilities ────────────────────────────────────────────────────────

def rot6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D continuous rotation → 3×3 rotation matrix via Gram-Schmidt.
    (..., 6)  →  (..., 3, 3)

    The first two columns of R are encoded in r6d[..., :3] and r6d[..., 3:6].
    Reference: Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019.
    """
    # Cast to float32: bfloat16 lacks precision for Gram-Schmidt (eps=1e-6 ≈ 0
    # in bf16, causing 0/0 NaN when a1 ∥ a2 at random initialisation).
    orig_dtype = r6d.dtype
    r6d = r6d.float()
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack as columns: R = [b1 | b2 | b3], restore original dtype
    return torch.stack([b1, b2, b3], dim=-1).to(orig_dtype)   # (..., 3, 3)

