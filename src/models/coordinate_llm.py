import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPredictionTransformer(nn.Module):
    """Two-layer transformer for per-patch 3D coordinate prediction.

    Architecture:
      1. Linear projection:  hidden_dim → d_model
      2. Add 2D sinusoidal positional encoding (variable grid size)
      3. Two TransformerEncoderLayer (pre-norm, batch_first)
      4. Linear projection:  d_model → 3 * upscale²
      5. PixelShuffle(upscale) → (h*upscale, w*upscale, 3)

    Sinusoidal 2D PE is regenerated on-the-fly for each (h, w), so the head
    handles variable-resolution images without any learned position parameters.
    """

    def __init__(
        self,
        hidden_dim:      int   = 2560,
        d_model:         int   = 512,
        nhead:           int   = 8,
        dim_feedforward: int   = 2048,
        upscale_factor:  int   = 4,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.d_model        = d_model

        self.input_proj  = nn.Linear(hidden_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj  = nn.Linear(d_model, 3 * (upscale_factor ** 2))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def _sinusoidal_2d_pe(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> torch.Tensor:
        """2D sinusoidal positional encoding.  Returns (h*w, d_model)."""
        half = self.d_model // 2          # row enc and col enc each use half dims

        # Shared frequency bands
        dim_idx = torch.arange(half, device=device, dtype=torch.float32)
        div     = torch.pow(10000.0, 2 * (dim_idx // 2) / half)  # (half,)

        row_idx = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)  # (h,1)
        col_idx = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)  # (w,1)

        row_enc = torch.zeros(h, half, device=device)
        row_enc[:, 0::2] = torch.sin(row_idx / div[0::2])
        row_enc[:, 1::2] = torch.cos(row_idx / div[1::2])

        col_enc = torch.zeros(w, half, device=device)
        col_enc[:, 0::2] = torch.sin(col_idx / div[0::2])
        col_enc[:, 1::2] = torch.cos(col_idx / div[1::2])

        # Broadcast and concat: (h, w, d_model)
        row_enc = row_enc.unsqueeze(1).expand(h, w, half)
        col_enc = col_enc.unsqueeze(0).expand(h, w, half)
        pe = torch.cat([row_enc, col_enc], dim=-1).reshape(h * w, self.d_model)
        return pe.to(dtype)

    def forward(
        self,
        hidden: torch.Tensor,   # (h*w, hidden_dim)
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Args:
            hidden:   (h*w, hidden_dim) — vision token hidden states
            h:        LLM patch grid height
            w:        LLM patch grid width
        Returns:
            (h*upscale, w*upscale, 3) — predicted xyz at sub-pixel resolution
        """
        x  = self.input_proj(hidden)                                    # (h*w, d_model)
        x  = x + self._sinusoidal_2d_pe(h, w, x.device, x.dtype)       # add 2D PE
        x  = self.transformer(x.unsqueeze(0)).squeeze(0)                # (h*w, d_model)
        x  = self.output_proj(x)                                        # (h*w, 3*up²)
        x  = x.view(1, h, w, -1).permute(0, 3, 1, 2)                   # (1, 3*up², h, w)
        x  = self.pixel_shuffle(x)                                      # (1, 3, h*up, w*up)
        return x[0].permute(1, 2, 0)                                    # (h*up, w*up, 3)


class CoordinateModel(nn.Module):
    """
    SpaForConditionalGeneration (+ LoRA) with two supervision signals:
      1. LM cross-entropy           → answer prediction loss (at answer tokens)
      2. DepthPredictionTransformer → xyz coordinate loss    (at vision tokens)

    The coord head reads LM hidden states at the <|image_pad|> vision-token
    positions directly; no dedicated per-patch text token is inserted.
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        coord_head:         DepthPredictionTransformer,
        image_token_id:     int,
        spatial_merge_size: int,
        skip_layers:        tuple[int, ...] = (-1,),
        answer_weight:      float = 1.0,
        coord_weight:       float = 1.0,
        polar:              bool  = False,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.coord_head         = coord_head
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.skip_layers        = list(skip_layers)
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight
        self.polar              = polar

        # Capture the probe hidden state via a hook, always. This avoids
        # `output_hidden_states=True`, which triggers a NaN bug on the
        # SpaDecTextModel (decouple) path with gradient checkpointing, and it
        # also preserves gradient-checkpointing memory savings.
        #
        # skip_layers[0] ==  -1  →  pre-hook on lm_head input (post-norm).
        # skip_layers[0] ==  -k  (k>=2) →  forward-hook on language_model
        #                                  .layers[-(k-1)] output (pre-norm).
        # In HF convention:
        #     hidden_states = (embeds, layer0_out, …, layer31_out, post_norm)
        #     → hidden_states[-k] == layers[-(k-1)].output   for k ≥ 2
        self._lm_head_input: torch.Tensor | None = None
        k = self.skip_layers[0]

        if k == -1:
            for name, mod in self.spa_model.named_modules():
                if name.endswith("lm_head"):
                    mod.register_forward_pre_hook(self._capture_lm_input)
                    break
        else:
            target_idx = k + 1  # e.g. -8 → -7 (layers[-7].output == hidden_states[-8])
            if target_idx >= 0:
                raise ValueError(
                    f"skip_layers[0]={k} must be negative (and ≤ -1); "
                    f"got target layer index {target_idx} which is non-negative."
                )
            layers = None
            for name, mod in self.spa_model.named_modules():
                if name.endswith("language_model.layers") and isinstance(mod, nn.ModuleList):
                    layers = mod
                    break
            if layers is None:
                raise RuntimeError(
                    "Could not find language_model.layers on spa_model for "
                    "the skip_layers forward-hook."
                )
            target = layers[target_idx]
            target.register_forward_hook(self._capture_layer_output)

    def _capture_lm_input(self, module, args):
        self._lm_head_input = args[0]

    def _capture_layer_output(self, module, inputs, output):
        # Decoder-layer forward may return a plain tensor or a tuple whose
        # first element is the hidden states.
        self._lm_head_input = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        input_ids:       torch.Tensor,           # (1, seq_len)
        attention_mask:  torch.Tensor,           # (1, seq_len)
        pixel_values:    torch.Tensor | None,    # (total_patches, C, H, W)
        image_grid_thw:  torch.Tensor | None,    # (num_images, 3)
        image_xyz:       list | None = None,
        image_xyz_hires: list | None = None,
        coord_scale          = 100.0,
        labels:          torch.Tensor | None = None,
        **kwargs,
    ):
        # ── backbone ─────────────────────────────────────────────────────────
        # Always use the hook-captured hidden state (set up in __init__):
        #   skip_layers[0] == -1  →  lm_head pre-hook (post-norm)
        #   skip_layers[0] == -k  →  layers[-(k-1)] forward-hook (pre-norm)
        # output_hidden_states=True path is intentionally avoided — it triggers
        # a NaN bug on the SpaDecTextModel (decouple) + gradient-checkpointing
        # path, and it would cost activation memory we don't need.
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = False,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            polar                = self.polar,
            **kwargs,
        )

        hidden_coord = self._lm_head_input

        logits = outputs.logits
        del outputs

        _ldict: dict = {}

        # ── LM answer-prediction loss ────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits.device)
            mask         = shift_labels[0] != -100
            _sl_m = shift_logits[0, mask]
            _sb_m = shift_labels[0, mask]
            lm_loss = F.cross_entropy(_sl_m, _sb_m)
            # Letter-position argmax (Option C from the paradigm-mismatch
            # postmortem). The supervised suffix is `<answer>{letter}</answer>`
            # plus `<|im_end|>\n`; the letter sits at index `letter_offset` in
            # the masked subset (= number of tokens in `<answer>`). Under
            # greedy decoding this matches evaluation.py's first-generated-
            # letter-token accuracy. Caller sets `model.letter_offset` once at
            # startup (see md/bug_fix/train_eval_paradigm_mismatch.md).
            if (not self.training) and _sl_m.numel() > 0:
                idx = getattr(self, "letter_offset", 0)
                if 0 <= idx < _sl_m.shape[0]:
                    _ldict["acc"] = (
                        1.0 if _sl_m[idx].argmax(-1).item() == int(_sb_m[idx].item())
                        else 0.0
                    )

        # ── per-patch 3D coordinate prediction ──────────────────────────────
        coord_loss = None
        coord_gt   = image_xyz_hires if image_xyz_hires is not None else image_xyz
        if coord_gt is not None and image_grid_thw is not None:
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
                if self.polar:
                    # (log r, θ=azimuth, α=inclination): weight angle channels
                    # by 1/π so a π-radian error matches a 1-unit log r error.
                    w = pred_k.new_tensor([1.0, 1.0 / math.pi, 1.0 / math.pi])
                    per_img_losses.append((pred_k - gt_k).abs().mul(w).mean())
                else:
                    per_img_losses.append(F.l1_loss(pred_k, gt_k))
                start += n_tok

            if per_img_losses:
                coord_loss = torch.stack(per_img_losses).mean()

        # ── combine losses ────────────────────────────────────────────────────
        loss = None
        if lm_loss is not None:
            _ldict["lm_loss"] = lm_loss.item()
            loss = self.answer_weight * lm_loss
        if coord_loss is not None:
            _ldict["coord_loss"] = coord_loss.item()
            loss = (loss + self.coord_weight * coord_loss) if loss is not None \
                   else (self.coord_weight * coord_loss)

        return None, loss, (_ldict if _ldict else None)
