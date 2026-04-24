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
        polar:       bool  = False,
        coord_scale: float = 100.0,
    ):
        super().__init__()
        self.spa_model   = spa_model
        self.use_xyz     = use_xyz
        self.polar       = polar
        self.coord_scale = coord_scale

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        image_xyz:      list | None = None,
        coord_scale:    float | None = None,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
        if coord_scale is None:
            coord_scale = self.coord_scale

        fwd_kwargs = dict(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = False,
            return_dict          = True,
        )
        if self.use_xyz:
            fwd_kwargs["image_xyz"]   = image_xyz
            fwd_kwargs["coord_scale"] = coord_scale
            if self.polar:
                fwd_kwargs["polar"] = True

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


class AnswerRelativeModel(nn.Module):
    """
    Like AnswerOnlyModel but for SpaRelativeForConditionalGeneration.

    Accepts ``image_xyz_relative`` (list of (N_frames, H, W, 3) tensors) and
    routes it to the backbone so per-query-frame M-RoPE can be applied.
    """

    def __init__(
        self,
        spa_model:   nn.Module,
        polar:       bool  = False,
        coord_scale: float = 100.0,
    ):
        super().__init__()
        self.spa_model   = spa_model
        self.polar       = polar
        self.coord_scale = coord_scale

    def forward(
        self,
        input_ids:           torch.Tensor,
        attention_mask:      torch.Tensor,
        pixel_values:        torch.Tensor | None,
        image_grid_thw:      torch.Tensor | None,
        image_xyz_relative:  list | None = None,
        coord_scale:         float | None = None,
        labels:              torch.Tensor | None = None,
        **kwargs,
    ):
        if coord_scale is None:
            coord_scale = self.coord_scale

        fwd_kwargs = dict(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = False,
            return_dict          = True,
        )
        if image_xyz_relative is not None:
            fwd_kwargs["image_xyz_relative"] = image_xyz_relative
            fwd_kwargs["coord_scale"]        = coord_scale
            if self.polar:
                fwd_kwargs["polar"] = True

        outputs = self.spa_model(**fwd_kwargs)

        if labels is None:
            return None, None, None

        logits       = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].to(logits.device)

        mask         = shift_labels[0] != -100
        shift_logits = shift_logits[0, mask]
        shift_labels = shift_labels[0, mask]
        lm_loss      = F.cross_entropy(shift_logits, shift_labels)
        return None, lm_loss, {"lm_loss": lm_loss.item()}

