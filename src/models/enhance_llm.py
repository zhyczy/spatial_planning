"""
EnhanceModel — frozen Qwen3.5-VL with sinusoidal 3D positional encoding
injected into each merged-patch vision token before the LLM consumes it.

Pipeline (per image k):
  1) Vision encoder produces e_img_k of shape (llm_h, llm_w, d) after the
     ViT merger.  llm_h = grid_thw[1] // spatial_merge_size, similarly W.
  2) Coordinate map c'_k of shape (llm_h, llm_w, 3) is supplied externally
     (already block-averaged from per-pixel xyz by the dataset's
     resize_xyz()).
  3) For each axis x, y, z separately, sinusoidal encoding to d_sub = d // 3
     dims using div_term = 10000^(2i/d_sub).  Concatenate along the feature
     dim and zero-pad to d if 3 * d_sub != d.
  4) e_vis_k = e_img_k + e_coord_k.  The LLM then sees position-aware
     visual embeddings WITHOUT any change to its own weights or to the
     standard 3D M-RoPE.

Trainable parameters: only `coord_head` (the auxiliary
DepthPredictionTransformer).  Qwen3.5-VL is fully frozen — vision encoder
and language model.  No LoRA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_3d_pe(coord_map: torch.Tensor, d: int) -> torch.Tensor:
    """
    Per-axis sinusoidal encoding of (x, y, z) coordinates.

    Args:
        coord_map: (..., 3) float tensor.  Last dim is (x, y, z).
        d:         target embedding dimension.

    Returns:
        (..., d) sinusoidal positional encoding.

    Construction (d_sub = d // 3, half = d_sub // 2):
        i        = [0, 1, ..., half - 1]
        div_term = 10000^(2i / d_sub)            (half,)

        For each axis a ∈ {x, y, z}:
            pe[..., a*d_sub + 2i]   = sin(coord_a / div_term[i])
            pe[..., a*d_sub + 2i+1] = cos(coord_a / div_term[i])

        If 3 * d_sub < d the trailing (d - 3*d_sub) dims are zero-padded.
    """
    if coord_map.shape[-1] != 3:
        raise ValueError(
            f"coord_map last dim must be 3, got {coord_map.shape[-1]}."
        )
    if d <= 0 or d < 6:
        raise ValueError(f"d must be >= 6, got {d}.")

    device = coord_map.device
    d_sub  = d // 3
    half   = d_sub // 2
    if half == 0:
        raise ValueError(f"d // 3 must be >= 2, got d_sub={d_sub} (d={d}).")

    cm = coord_map.to(torch.float32)                    # (..., 3)

    i        = torch.arange(half, device=device, dtype=torch.float32)
    div_term = torch.pow(10000.0, (2.0 * i) / float(d_sub))   # (half,)

    out_shape = (*cm.shape[:-1], d)
    pe = cm.new_zeros(out_shape, dtype=torch.float32)
    # 2*half positions filled per axis; trailing slots stay zero (this
    # matters when d_sub is odd, since half = d_sub // 2).
    used = 2 * half

    for axis in range(3):
        scaled = cm[..., axis : axis + 1] / div_term         # (..., half)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        offset = axis * d_sub
        pe[..., offset : offset + used : 2] = s
        pe[..., offset + 1 : offset + used : 2] = c

    return pe


class EnhanceModel(nn.Module):
    """
    Frozen Qwen3.5-VL + sinusoidal 3D PE on merged-patch vision tokens.

    Auxiliary DepthPredictionTransformer (`coord_head`) reads vision-token
    hidden states from the LLM and predicts sub-pixel xyz; this is the
    only trainable component.

    Forward signature mirrors CoordinateModel for drop-in interchange in
    the training loop.
    """

    def __init__(
        self,
        base_model:         nn.Module,            # Qwen3_5ForConditionalGeneration
        coord_head:         nn.Module | None,
        image_token_id:     int,
        spatial_merge_size: int,
        skip_layers:        tuple[int, ...] = (-1,),
        answer_weight:      float = 1.0,
        coord_weight:       float = 1.0,
    ):
        super().__init__()
        self.base_model         = base_model
        self.coord_head         = coord_head
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.skip_layers        = list(skip_layers)
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight

        # ── hidden-state capture for coord_head (optional) ──────────────
        # Same convention as CoordinateModel:
        #   skip_layers[0] == -1 → pre-hook on lm_head input (post-norm)
        #   skip_layers[0] == -k → forward-hook on language_model.layers[-(k-1)]
        self._lm_head_input: torch.Tensor | None = None
        if coord_head is not None:
            k = self.skip_layers[0]
            if k == -1:
                for name, mod in self.base_model.named_modules():
                    if name.endswith("lm_head"):
                        mod.register_forward_pre_hook(self._capture_lm_input)
                        break
            else:
                target_idx = k + 1
                if target_idx >= 0:
                    raise ValueError(
                        f"skip_layers[0]={k} must be negative; got "
                        f"target index {target_idx}."
                    )
                layers = None
                for name, mod in self.base_model.named_modules():
                    if name.endswith("language_model.layers") and isinstance(
                        mod, nn.ModuleList
                    ):
                        layers = mod
                        break
                if layers is None:
                    raise RuntimeError(
                        "Could not find language_model.layers on base_model."
                    )
                layers[target_idx].register_forward_hook(self._capture_layer_output)

    def _capture_lm_input(self, module, args):
        self._lm_head_input = args[0]

    def _capture_layer_output(self, module, inputs, output):
        self._lm_head_input = output[0] if isinstance(output, tuple) else output

    # ── PE injection helper ────────────────────────────────────────────
    def _inject_3d_pe(
        self,
        image_embeds_tuple: tuple[torch.Tensor, ...],
        image_grid_thw:     torch.Tensor,
        image_xyz:          list | None,
    ) -> list[torch.Tensor]:
        """
        Per-image sinusoidal 3D PE addition.

        image_embeds_tuple[k]: (llm_h_k * llm_w_k, hidden_dim) — already
                                merged patches from the ViT merger.
        image_grid_thw[k]:     (T, H, W) in patch units (pre-merge).
        image_xyz[k]:          (llm_h_k, llm_w_k, 3) float coordinates.
        """
        sms = self.spatial_merge_size
        out: list[torch.Tensor] = []
        for k, embeds_k in enumerate(image_embeds_tuple):
            if image_xyz is None or k >= len(image_xyz):
                out.append(embeds_k)
                continue

            thw_k  = image_grid_thw[k]
            llm_h  = int(thw_k[1]) // sms
            llm_w  = int(thw_k[2]) // sms
            d      = embeds_k.shape[-1]

            coord_k = image_xyz[k].to(embeds_k.device)
            if coord_k.shape[0] != llm_h or coord_k.shape[1] != llm_w:
                # Mismatch — dataset shape disagrees with grid_thw. Skip
                # PE for this image rather than silently corrupt features.
                out.append(embeds_k)
                continue

            pe = sinusoidal_3d_pe(coord_k, d=d)            # (llm_h, llm_w, d)
            pe = pe.reshape(-1, d).to(embeds_k.dtype)      # (llm_h*llm_w, d)
            out.append(embeds_k + pe)
        return out

    # ── forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        pixel_values:    torch.Tensor | None,
        image_grid_thw:  torch.Tensor | None,
        image_xyz:       list | None = None,
        image_xyz_hires: list | None = None,
        coord_scale          = 100.0,
        labels:          torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        bm = self.base_model.model  # Qwen3_5Model

        # ── 1) text token embeddings ───────────────────────────────────
        inputs_embeds = bm.get_input_embeddings()(input_ids)

        # ── 2) vision encoder + 3D PE injection ────────────────────────
        if pixel_values is not None:
            with torch.no_grad():
                image_outputs = bm.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True,
                )
            image_embeds_tuple = image_outputs.pooler_output  # tuple per image

            new_embeds = self._inject_3d_pe(
                image_embeds_tuple, image_grid_thw, image_xyz,
            )

            image_embeds = torch.cat(new_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask, _ = bm.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # ── 3) 3D M-RoPE position ids (Qwen original) ──────────────────
        # Qwen3_5Model.compute_3d_position_ids requires mm_token_type_ids.
        # Derive it from input_ids if the caller didn't supply one:
        #   1 = image token, 0 = text. (Videos unsupported here.)
        if mm_token_type_ids is None:
            mm_token_type_ids = (
                input_ids == self.image_token_id
            ).to(torch.int32)
        position_ids = bm.compute_3d_position_ids(
            input_ids        = input_ids,
            inputs_embeds    = inputs_embeds,
            image_grid_thw   = image_grid_thw,
            video_grid_thw   = None,
            attention_mask   = attention_mask,
            past_key_values  = None,
            mm_token_type_ids= mm_token_type_ids,
        )

        # ── 4) language model + lm_head ────────────────────────────────
        # Qwen weights are frozen so requires_grad chain is broken at
        # inputs_embeds. The forward still computes correct activations;
        # coord_head sees them via the hidden-state hook.
        lm_outputs = bm.language_model(
            input_ids        = None,
            inputs_embeds    = inputs_embeds,
            attention_mask   = attention_mask,
            position_ids     = position_ids,
            return_dict      = True,
        )
        logits = self.base_model.lm_head(lm_outputs.last_hidden_state)

        hidden_coord = self._lm_head_input

        _ldict: dict = {}

        # ── 5) LM cross-entropy (logging only — Qwen frozen) ───────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits.device)
            mask         = shift_labels[0] != -100
            _sl_m = shift_logits[0, mask]
            _sb_m = shift_labels[0, mask]
            if _sl_m.numel() > 0:
                lm_loss = F.cross_entropy(_sl_m, _sb_m)
                if (not self.training):
                    idx = getattr(self, "letter_offset", 0)
                    if 0 <= idx < _sl_m.shape[0]:
                        _ldict["acc"] = (
                            1.0 if _sl_m[idx].argmax(-1).item() == int(_sb_m[idx].item())
                            else 0.0
                        )

        # ── 6) per-patch coord loss (the only trainable signal) ────────
        coord_loss = None
        coord_gt   = image_xyz_hires if image_xyz_hires is not None else image_xyz
        if (
            self.coord_head is not None
            and coord_gt is not None
            and image_grid_thw is not None
            and hidden_coord is not None
        ):
            vis_pos = (input_ids[0] == self.image_token_id).nonzero(
                as_tuple=True
            )[0]

            sms   = self.spatial_merge_size
            start = 0
            per_img_losses: list[torch.Tensor] = []

            for k in range(min(len(coord_gt), len(image_grid_thw))):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // sms
                llm_w = int(thw_k[2]) // sms
                n_tok = llm_h * llm_w

                if start + n_tok > len(vis_pos):
                    break

                coord_h_k = hidden_coord[0, vis_pos[start : start + n_tok]]
                pred_k    = self.coord_head(coord_h_k, llm_h, llm_w)

                gt_k = coord_gt[k].to(pred_k.device, dtype=pred_k.dtype)
                per_img_losses.append(F.l1_loss(pred_k, gt_k))
                start += n_tok

            if per_img_losses:
                coord_loss = torch.stack(per_img_losses).mean()

        # ── 7) combine ─────────────────────────────────────────────────
        loss = None
        if lm_loss is not None:
            _ldict["lm_loss"] = lm_loss.item()
            # Scale by 0 since Qwen is frozen — keep value for logging
            # but contribute zero to the optimised loss.
            if lm_loss.requires_grad:
                loss = self.answer_weight * lm_loss
        if coord_loss is not None:
            _ldict["coord_loss"] = coord_loss.item()
            loss = (
                (loss + self.coord_weight * coord_loss) if loss is not None
                else (self.coord_weight * coord_loss)
            )

        return None, loss, (_ldict if _ldict else None)
