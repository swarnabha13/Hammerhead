"""
run_all.py — Train + Evaluate + Plot (swing-up AND balance)
============================================================
    python run_all.py
    python run_all.py --total-timesteps 3000000 --rand-range 0.20
    python run_all.py --generate-gifs
"""

from __future__ import annotations
import argparse, os, subprocess, sys
from train    import train, TrainConfig
from evaluate import evaluate_all, _plot_training_curve, MISMATCH_LEVELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps",   type=int,   default=2_000_000)
    p.add_argument("--rand-range",        type=float, default=0.20)
    p.add_argument("--max-episode-steps", type=int,   default=1000)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--n-eval-episodes",   type=int,   default=50)
    p.add_argument("--save-dir",          type=str,   default="./results")
    p.add_argument("--cuda",              action="store_true")
    p.add_argument("--generate-gifs",     action="store_true",
                   help="Also render GIFs after evaluation (requires pygame)")
    a = p.parse_args()

    cfg = TrainConfig(
        seed              = a.seed,
        cuda              = a.cuda,
        rand_range        = a.rand_range,
        total_timesteps   = a.total_timesteps,
        max_episode_steps = a.max_episode_steps,
        save_dir          = a.save_dir,
    )
    train(cfg)

    history_path = os.path.join(a.save_dir, "training_history.json")
    curve_path   = _plot_training_curve(history_path, a.save_dir)
    print(f"  Training curve → {curve_path}")

    model_path = os.path.join(a.save_dir, "ppo_acrobot.pth")
    evaluate_all(
        model_path      = model_path,
        n_episodes      = a.n_eval_episodes,
        seed            = a.seed,
        save_dir        = a.save_dir,
        mismatch_levels = MISMATCH_LEVELS,
    )

    if a.generate_gifs:
        print("\n  Generating GIFs (requires pygame) ...")
        subprocess.run([sys.executable, "visualize.py",
                        "--model-path", model_path,
                        "--levels", "-0.2,-0.1,0.0,0.1,0.2",
                        "--n-tries", "3"], check=False)

    print("\n  ✓ All done. See results/ and gifs/ for outputs.\n")


if __name__ == "__main__":
    main()
