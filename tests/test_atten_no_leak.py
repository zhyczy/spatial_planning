"""
Leak regression test for the atten attention path.

Background: see md/bug_fix/train_eval_paradigm_mismatch.md §5.

The smoking gun: train_atten.py reports 96 % first-token argmax accuracy in
TF eval, but offline `model.generate()` produces non-letter first token
(observed: 1050/1050 outputs start with "T"). Under greedy decoding these two
quantities are mathematically identical (same weights, same context). So one
of them is wrong → forward path is leaking.

This test takes ONE MindCube sample and computes:

  full_ids   = prompt + answer + <|im_end|> + \\n         (length L)
  prompt_ids = prompt                                    (length L_p)

  out_tf  = model(full_ids)              # TF forward
  out_gen = model(prompt_ids)            # gen prefill

  tf_argmax  = out_tf.logits[0, L_p-1, :].argmax()
  gen_argmax = out_gen.logits[0, L_p-1, :].argmax()

These MUST be equal. If not, a leak.

Usage
-----
CUDA_VISIBLE_DEVICES=0 \\
python tests/test_atten_no_leak.py \\
    --ckpt        train_records/app/atten_mindcube/step_1000 \\
    --model_path  checkpoints/Qwen3.5-4B \\
    --data_dir    datasets/evaluation/MindCube
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
while str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))

from evaluation import load_spa_model, prepare_batch_spa
from src.dataset import load_testing_dataset


def run_forward(spa, proc, inputs_dev, image_xyz_dev, mm_type_ids):
    """Single forward pass through the atten model.

    image_xyz is stashed on spa.model._eval_image_xyz (per the shim installed
    in load_spa_model), and we explicitly hand mm_token_type_ids to the
    forward — same plumbing as evaluation.py's prefill path.
    """
    spa.model._eval_image_xyz = image_xyz_dev
    with torch.no_grad():
        kwargs = dict(
            input_ids       = inputs_dev["input_ids"],
            attention_mask  = inputs_dev["attention_mask"],
            mm_token_type_ids = mm_type_ids,
        )
        if "pixel_values" in inputs_dev and inputs_dev["pixel_values"] is not None:
            kwargs["pixel_values"]   = inputs_dev["pixel_values"]
        if "image_grid_thw" in inputs_dev and inputs_dev["image_grid_thw"] is not None:
            kwargs["image_grid_thw"] = inputs_dev["image_grid_thw"]
        out = spa(**kwargs)
    spa.model._eval_image_xyz = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", default="mindcube")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device

    # ── Load model ───────────────────────────────────────────────────────────
    print("=== Loading atten model …")
    spa, proc = load_spa_model(
        args.model_path, args.ckpt, device,
        vanilla=False, decouple=False, atten=True,
    )
    spa.eval()

    # ── Load samples ─────────────────────────────────────────────────────────
    ds = load_testing_dataset(
        Path(args.data_dir).resolve(), limit=args.n_samples, dataset=args.dataset,
    )
    print(f"=== Loaded {len(ds)} sample(s)")

    # ── Per-sample comparison ────────────────────────────────────────────────
    n_leaks = 0
    for i, item in enumerate(ds):
        print(f"\n──────── sample {i} (idx={item.get('index')}) ────────")
        gt_letter = item.get("answer", "").strip().upper()
        print(f"  GT answer: {gt_letter!r}")

        # Build gen-style inputs (prompt only, ends at "<|im_start|>assistant\n")
        gen_inputs, _, image_xyz = prepare_batch_spa(
            item, proc, spatial_merge_size=2, use_coord=False, coord_scale=100.0,
            thinking=False, load_xyz=True, strict_xyz=True,
        )
        # Move to device
        gen_inputs_dev = {}
        for k, v in gen_inputs.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    gen_inputs_dev[k] = v.to(device, dtype=torch.bfloat16)
                else:
                    gen_inputs_dev[k] = v.to(device)
            else:
                gen_inputs_dev[k] = v
        xyz_dev = [x.to(device) for x in image_xyz]

        L_p = gen_inputs_dev["input_ids"].shape[1]

        # Build TF-style inputs (prompt + letter + <|im_end|> + "\n")
        suffix_text = gt_letter + "<|im_end|>\n"
        suffix_ids = proc.tokenizer(
            suffix_text, add_special_tokens=False
        )["input_ids"]
        suffix_ids_t = torch.tensor([suffix_ids], device=device, dtype=torch.long)
        L_suffix = suffix_ids_t.shape[1]

        tf_input_ids = torch.cat([gen_inputs_dev["input_ids"], suffix_ids_t], dim=1)
        tf_attn      = torch.cat([
            gen_inputs_dev["attention_mask"],
            torch.ones_like(suffix_ids_t),
        ], dim=1)
        tf_mm        = torch.cat([
            gen_inputs_dev["mm_token_type_ids"],
            torch.zeros_like(suffix_ids_t),     # answer is text (type 0)
        ], dim=1)

        tf_inputs_dev = dict(gen_inputs_dev)
        tf_inputs_dev["input_ids"]      = tf_input_ids
        tf_inputs_dev["attention_mask"] = tf_attn

        print(f"  L_p (prompt only) = {L_p}")
        print(f"  L_suffix (letter + im_end + \\n) = {L_suffix}  → {suffix_ids}")
        print(f"  L_TF = {tf_input_ids.shape[1]}")

        # ── Forward both ─────────────────────────────────────────────────────
        out_gen = run_forward(spa, proc, gen_inputs_dev, xyz_dev,
                              gen_inputs_dev["mm_token_type_ids"])
        out_tf  = run_forward(spa, proc, tf_inputs_dev, xyz_dev, tf_mm)

        # ── Compare logits at position L_p - 1 ───────────────────────────────
        gen_logits = out_gen.logits[0, L_p - 1, :].float()
        tf_logits  = out_tf.logits [0, L_p - 1, :].float()

        gen_argmax = gen_logits.argmax(-1).item()
        tf_argmax  = tf_logits.argmax(-1).item()
        diff       = (tf_logits - gen_logits).abs().max().item()
        cossim     = torch.nn.functional.cosine_similarity(
            tf_logits[None, :], gen_logits[None, :]
        ).item()

        gen_tok = proc.tokenizer.decode([gen_argmax])
        tf_tok  = proc.tokenizer.decode([tf_argmax])
        gt_tok  = proc.tokenizer.decode([suffix_ids[0]])

        print(f"  GT first token (letter)        : id={suffix_ids[0]:>6d}  decoded={gt_tok!r}")
        print(f"  Gen prefill argmax             : id={gen_argmax:>6d}  decoded={gen_tok!r}")
        print(f"  TF forward  argmax             : id={tf_argmax:>6d}  decoded={tf_tok!r}")
        print(f"  max |tf_logits - gen_logits|   : {diff:.6e}")
        print(f"  cos(tf_logits, gen_logits)     : {cossim:.6f}")

        if tf_argmax == gen_argmax and diff < 1e-3:
            print(f"  → OK: TF and Gen identical (no leak observed at this position)")
        else:
            print(f"  → LEAK: TF != Gen; this position is forward-path-leaky")
            n_leaks += 1

            # Top-5 from each side (compare ranks)
            gen_top5 = gen_logits.topk(5)
            tf_top5  = tf_logits.topk(5)
            print(f"     gen top-5: {[proc.tokenizer.decode([t.item()]) for t in gen_top5.indices]}")
            print(f"     tf  top-5: {[proc.tokenizer.decode([t.item()]) for t in tf_top5.indices]}")

    print(f"\n=== summary: {n_leaks}/{len(ds)} samples show leak (TF != Gen at L_p-1)")
    if n_leaks > 0:
        print("    → leak in atten attention path; see md/bug_fix/train_eval_paradigm_mismatch.md §6.1 for entry candidates A/B/C")
        sys.exit(1)
    else:
        print("    → no leak detected (TF == Gen everywhere); 96% TF eval gap has another explanation")


if __name__ == "__main__":
    main()
