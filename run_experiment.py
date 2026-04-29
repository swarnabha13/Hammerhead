"""
run_experiment.py — Full pipeline: train → evaluate → visualize.

Usage:
  python run_experiment.py                  # full 2M-step run (~20-40 min)
  python run_experiment.py --quick          # 300k steps smoke-test (~3-5 min)
  python run_experiment.py --eval-only --checkpoint checkpoints/NAME_best.pt
"""
from __future__ import annotations
import argparse, glob, os, sys, time

# Small stage wrapper so long runs show where time is being spent.
def run_stage(label: str, fn, *args, **kwargs):
    print(f"\n{'='*60}\nSTAGE: {label}\n{'='*60}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    return result

def find_latest_checkpoint() -> str:
    # Useful for eval-only reruns after training several checkpoints.
    ckpts = glob.glob("checkpoints/*.pt")
    if not ckpts:
        raise FileNotFoundError("No checkpoints found. Run training first.")
    return max(ckpts, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",     action="store_true", help="Short training run for testing")
    parser.add_argument("--eval-only", action="store_true", help="Skip training")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--num-envs",   type=int, default=8)
    args = parser.parse_args()

    if not args.eval_only:
        from train import Args, train
        train_args = Args()
        train_args.seed              = args.seed
        train_args.num_envs          = args.num_envs
        train_args.total_timesteps   = 300_000 if args.quick else 2_000_000
        train_args.num_steps         = 128     if args.quick else 256
        train_args.dr_range          = 0.20
        train_args.max_episode_steps = 500
        checkpoint = run_stage("TRAINING", train, train_args)
    else:
        checkpoint = args.checkpoint or find_latest_checkpoint()

    print(f"\nUsing checkpoint: {checkpoint}")

    from evaluate import run_evaluation, print_summary
    n_ep = 20 if args.quick else args.n_episodes
    df = run_stage("EVALUATION", run_evaluation,
                   checkpoint_path=checkpoint,
                   n_episodes=n_ep,
                   seed=args.seed,
                   n_random_seeds=3 if args.quick else 5)
    print_summary(df)

    from visualize import main as viz_main
    run_stage("VISUALISATION", viz_main, "results/eval_results.csv")

    print("\n✓ Experiment complete.")
    print("  Results CSV : results/eval_results.csv")
    print("  Plots       : results/*.png")
    print("  TensorBoard : tensorboard --logdir runs/")

if __name__ == "__main__":
    main()
