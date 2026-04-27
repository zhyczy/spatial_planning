"""
Top-level orchestrator for the 3DRS baseline adapted to MindCube / SPINBench.

Wraps the shell scripts under `repo/3DRS/scripts/mindcube/` so you can launch
feature extraction, training, or evaluation from a single Python entry point
(matching how other spatial_planning baselines are driven).

Typical usage:
    python -m spatial_planning.baseline.3drs extract_vggt
    python -m spatial_planning.baseline.3drs train
    python -m spatial_planning.baseline.3drs eval_tinybench --ckpt ./ckpt/llavanext-qwen-3drs-mindcube
    python -m spatial_planning.baseline.3drs eval_spinbench --ckpt ./ckpt/llavanext-qwen-3drs-mindcube

Architecture notes:
    * Dataset layout: reuses the same per-sample 3d_results roots that
      train_alternate.py / evaluation.py already produce, so there's no second
      copy of data.
    * VGGT features (`vggt.npz`) are pre-extracted once per sample with a
      fixed (392, 518) input so every sample yields L=1036 patches (one of
      the supported shapes in feature_3d_alignment).
    * Training runs through `llava.train.train_mindcube`, which monkey-patches
      `LazySupervisedDataset` -> `MindcubeDataset` before calling the upstream
      `train()` so the existing deepspeed/torchrun machinery is unchanged.
    * Only the scanqa-like QA branch is exercised; grounding / box_input /
      box_label code paths are never hit (`--ground_head_type None`,
      `metadata.dataset == "scanqa"`).
"""
import argparse
import os
import subprocess
import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo", "3DRS")
SCRIPTS = os.path.join(REPO_DIR, "scripts", "mindcube")

STAGES = {
    "extract_vggt": "extract_vggt.sh",
    "train": "train.sh",
    "eval_tinybench": "eval_tinybench.sh",
    "eval_spinbench": "eval_spinbench.sh",
}


def _run(script_name, extra_env=None):
    path = os.path.join(SCRIPTS, script_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[3drs] running {path}", flush=True)
    return subprocess.call(["bash", path], cwd=REPO_DIR, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES.keys()))
    parser.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint dir for eval stages (exports $CKPT)",
    )
    parser.add_argument(
        "--answers",
        default=None,
        help="answer jsonl path for eval stages (exports $ANSWERS)",
    )
    parser.add_argument(
        "--n-gpu", type=int, default=None, help="override $N_GPU / $NUM_GPUS"
    )
    args = parser.parse_args()

    extra = {}
    if args.ckpt:
        extra["CKPT"] = args.ckpt
    if args.answers:
        extra["ANSWERS"] = args.answers
    if args.n_gpu is not None:
        extra["N_GPU"] = str(args.n_gpu)
        extra["NUM_GPUS"] = str(args.n_gpu)

    rc = _run(STAGES[args.stage], extra_env=extra)
    sys.exit(rc)


if __name__ == "__main__":
    main()
