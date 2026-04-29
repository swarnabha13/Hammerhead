"""
Curriculum launcher for full down-start swing-up plus upright hold.

This script intentionally lives outside the main training path. It reuses
train.train() but runs several staged jobs:

1. Start from an older PPO checkpoint that can swing the Acrobot high.
2. Mix in near-upright reset curriculum so the policy keeps the DAgger hold skill.
3. Gradually reduce balance resets until all episodes start from the default down state.

The goal is a single policy that can start down, swing up, and then remain in
the strict balance region. This is experimental; the validated hold controller
and its results are left unchanged.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import train  # noqa: E402


def build_args(
    *,
    exp_name: str,
    resume_checkpoint: str | None,
    dr_range: float,
    balance_reset_prob: float,
    total_timesteps: int,
    teacher_action_prob: float,
    teacher_final_prob: float,
    bc_coef: float,
    bc_final_coef: float,
    seed: int,
    num_envs: int,
    save_dir: str,
) -> SimpleNamespace:
    num_steps = 256
    num_minibatches = 8
    batch_size = num_envs * num_steps
    return SimpleNamespace(
        exp_name=exp_name,
        seed=seed,
        cuda=False,
        track=False,
        resume_checkpoint=resume_checkpoint,
        dr_range=dr_range,
        num_envs=num_envs,
        balance_reset_prob=balance_reset_prob,
        total_timesteps=total_timesteps,
        learning_rate=1.0e-4,
        num_steps=num_steps,
        anneal_lr=True,
        gamma=0.995,
        gae_lambda=0.95,
        num_minibatches=num_minibatches,
        update_epochs=6,
        clip_coef=0.1,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        bc_coef=bc_coef,
        bc_final_coef=bc_final_coef,
        teacher_action_prob=teacher_action_prob,
        teacher_final_prob=teacher_final_prob,
        pretrain_bc_steps=0,
        pretrain_bc_batch_size=256,
        pretrain_bc_lr=1.0e-3,
        pretrain_eval_interval=100,
        pretrain_reset_fraction=0.10,
        pretrain_policy_fraction=0.85,
        pretrain_reset_on_fall=True,
        save_dir=save_dir,
        batch_size=batch_size,
        minibatch_size=batch_size // num_minibatches,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run staged full swing-up curriculum")
    p.add_argument(
        "--swingup-checkpoint",
        default=os.path.join(
            "checkpoints",
            "ppo_robust_dr5_balance500_dr0.05_seed42_1777321406.pt",
        ),
        help="Checkpoint used to initialize the swing-up behavior",
    )
    p.add_argument("--exp-prefix", default="full_swingup_curriculum")
    p.add_argument("--save-dir", default=os.path.join("checkpoints", "full_swingup"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--stage-steps", type=int, default=1_000_000)
    p.add_argument("--dr-range", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    stages = [
        # Keep many near-upright starts early so PPO does not forget the hold basin.
        dict(balance_reset_prob=0.70, teacher_action_prob=0.35, bc_coef=1.0),
        # Force more down-start and mid-swing recovery.
        dict(balance_reset_prob=0.35, teacher_action_prob=0.25, bc_coef=0.8),
        # Mostly down-start, still with a little capture-state teacher pressure.
        dict(balance_reset_prob=0.10, teacher_action_prob=0.15, bc_coef=0.6),
        # Final down-start-only policy optimization.
        dict(balance_reset_prob=0.00, teacher_action_prob=0.05, bc_coef=0.4),
    ]

    ckpt = args.swingup_checkpoint
    print("\nFull swing-up curriculum")
    print(f"  Initial checkpoint: {ckpt}")
    print(f"  Stage steps       : {args.stage_steps:,}")
    print(f"  Save dir          : {args.save_dir}")

    for idx, cfg in enumerate(stages, start=1):
        stage_name = f"{args.exp_prefix}_s{idx}"
        print("\n" + "=" * 72)
        print(f"Starting stage {idx}/{len(stages)}: {stage_name}")
        print(f"  resume              : {ckpt}")
        print(f"  balance_reset_prob  : {cfg['balance_reset_prob']}")
        print(f"  teacher_action_prob : {cfg['teacher_action_prob']}")
        print(f"  bc_coef             : {cfg['bc_coef']}")
        print("=" * 72)

        train_args = build_args(
            exp_name=stage_name,
            resume_checkpoint=ckpt,
            dr_range=args.dr_range,
            balance_reset_prob=cfg["balance_reset_prob"],
            total_timesteps=args.stage_steps,
            teacher_action_prob=cfg["teacher_action_prob"],
            teacher_final_prob=0.0,
            bc_coef=cfg["bc_coef"],
            bc_final_coef=0.25,
            seed=args.seed,
            num_envs=args.num_envs,
            save_dir=args.save_dir,
        )
        ckpt = train(train_args)

    print("\nCurriculum complete.")
    print(f"Final checkpoint: {ckpt}")
    print("\nEvaluate down-start performance with:")
    print(
        "python -B evaluate.py "
        f"--checkpoint \"{ckpt}\" "
        "--controller policy --balance-reset-prob 0.0 "
        "--num-episodes 100 --label full_swingup_policy"
    )
    print("\nRender a down-start GIF with:")
    print(
        "python -B render_gif.py "
        f"--checkpoint \"{ckpt}\" "
        "--controller policy --balance-reset-prob 0.0 "
        "--mismatches 0.0 --duration 42 --fps 24 --seed 1 --no-comparison"
    )


if __name__ == "__main__":
    main()
