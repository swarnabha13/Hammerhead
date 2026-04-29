# Full Down-Start Swing-Up Experiment

The main project currently contains a validated robust upright-hold controller.
That controller works from near-upright starts, but it is not a full down-start
swing-up controller.

This folder contains an experimental curriculum for learning the missing part:

```text
default down start -> swing up -> enter capture region -> hold upright
```

The code here is separate from the validated hold-controller workflow so the
existing results remain intact.

## Why a New Experiment Is Needed

The current best hold checkpoint:

```text
checkpoints/dagger_hold_dr10_dr0.1_seed42_1777437143.pt
```

performs well when evaluation starts near upright:

```text
--balance-reset-prob 1.0
```

but it does not solve down-start swing-up:

```text
--balance-reset-prob 0.0
```

An older PPO checkpoint can swing the tip high, but it does not slow the system
enough to satisfy the strict balance condition. The experiment here starts from
that older swing-up checkpoint and then gradually shifts the reset distribution
toward the default down start.

## Run the Curriculum

From the repository root:

```powershell
python -B experiments\full_swingup\train_full_swingup_curriculum.py `
  --swingup-checkpoint checkpoints\ppo_robust_dr5_balance500_dr0.05_seed42_1777321406.pt `
  --exp-prefix full_swingup_curriculum `
  --save-dir checkpoints\full_swingup `
  --stage-steps 1000000 `
  --dr-range 0.0 `
  --num-envs 16 `
  --seed 42
```

The launcher runs four stages:

| Stage | Balance reset probability | Teacher action probability | Purpose |
|---:|---:|---:|---|
| 1 | 0.70 | 0.35 | Preserve hold behavior while starting from swing-up weights |
| 2 | 0.35 | 0.25 | Increase down-start pressure |
| 3 | 0.10 | 0.15 | Mostly down-start training |
| 4 | 0.00 | 0.05 | Final down-start-only policy optimization |

The final checkpoint path is printed at the end.

## Evaluate

Use down-start evaluation:

```powershell
python -B evaluate.py `
  --checkpoint checkpoints\full_swingup\YOUR_FINAL_CHECKPOINT.pt `
  --controller policy `
  --balance-reset-prob 0.0 `
  --num-episodes 100 `
  --label full_swingup_policy
```

The run is useful only if nominal `+0%` has nonzero sustained hold:

```text
HoldMax > 100   minimum useful result
HoldMax > 300   good result
HoldMax > 500   strong result
```

## Render GIF

Only render after evaluation confirms nonzero down-start hold:

```powershell
python -B render_gif.py `
  --checkpoint checkpoints\full_swingup\YOUR_FINAL_CHECKPOINT.pt `
  --controller policy `
  --balance-reset-prob 0.0 `
  --mismatches 0.0 `
  --duration 42 `
  --fps 24 `
  --seed 1 `
  --no-comparison
```

If the rendered GIF reports `Max hold 0`, try a few seeds, but do not report a
successful swing-up result unless evaluation over many episodes also shows
nonzero hold.
