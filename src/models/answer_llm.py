import torch
import torch.nn as nn
import torch.nn.functional as F

class AnswerOnlyModel(nn.Module):
    """
    Ablation model: LoRA fine-tuning with only LM answer-prediction loss.
    No pose regression head, no <pose> tokens.

    Used for two ablation studies:
      - no_cam:  keeps 4D M-RoPE (image_xyz passed), removes pose prediction
      - vanilla: uses original 3D M-RoPE (no image_xyz), removes pose prediction
    """

    def __init__(self, spa_model: nn.Module, use_xyz: bool = True):
        super().__init__()
        self.spa_model = spa_model
        self.use_xyz = use_xyz

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        image_xyz:      list | None = None,
        coord_scale:    float = 100.0,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
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

