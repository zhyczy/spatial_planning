import torch
import torch.nn as nn
import torch.nn.functional as F

class AnswerOnlyModel(nn.Module):
    """
    LoRA fine-tuning with only the LM answer-prediction loss.

    Two modes:
      - no_cam:  keeps 4D M-RoPE (image_xyz passed)
      - vanilla: uses original 3D M-RoPE (no image_xyz)
    """

    def __init__(
        self,
        spa_model:   nn.Module,
        use_xyz:     bool  = True,
        coord_scale: float = 100.0,
    ):
        super().__init__()
        self.spa_model   = spa_model
        self.use_xyz     = use_xyz
        self.coord_scale = coord_scale

    def forward(
        self,
        input_ids:         torch.Tensor,
        attention_mask:    torch.Tensor,
        pixel_values:      torch.Tensor | None,
        image_grid_thw:    torch.Tensor | None,
        image_xyz:         list | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        coord_scale:       float | None = None,
        labels:            torch.Tensor | None = None,
        **kwargs,
    ):
        if coord_scale is None:
            coord_scale = self.coord_scale

        fwd_kwargs = dict(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            # mm_token_type_ids must reach Qwen3_5Model — it's also what
            # SpatialAttnVanillaModel.forward needs to build _spatial_cache.
            # Without it, vision_mask cannot be derived → bias module dead.
            mm_token_type_ids    = mm_token_type_ids,
            output_hidden_states = False,
            return_dict          = True,
        )
        if self.use_xyz:
            fwd_kwargs["image_xyz"]   = image_xyz
            fwd_kwargs["coord_scale"] = coord_scale

        outputs = self.spa_model(**fwd_kwargs)

        if labels is None:
            return None, None, None

        logits = outputs.logits                          # (1, seq_len, V)
        shift_logits = logits[:, :-1, :]                   # (1, seq_len-1, V)
        shift_labels = labels[:, 1:].to(logits.device)     # (1, seq_len-1)

        # 只取有效 label 位置的 logits，避免整个 (seq_len, V) 留在显存里
        mask = shift_labels[0] != -100                     # (seq_len-1,)
        shift_logits = shift_logits[0, mask]               # (N_valid, V)
        shift_labels = shift_labels[0, mask]               # (N_valid,)
        lm_loss = F.cross_entropy(shift_logits, shift_labels)
        _ldict = {"lm_loss": lm_loss.item()}
        return None, lm_loss, _ldict
