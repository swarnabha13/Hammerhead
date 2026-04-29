# Robust Acrobot Upright Hold

This repository trains and evaluates an Acrobot controller that holds both links near the upright position under simulator parameter mismatch.

The project is based on the local assignment brief: [Problem statement.pdf](Problem%20statement.pdf). The brief asks for a reinforcement-learning solution around Gymnasium Acrobot, CleanRL-style implementation, and robustness to dynamics mismatch. The final implementation keeps PPO infrastructure in the codebase, but the successful controller reported here is trained with DAgger-style imitation learning from a stabilizing teacher, followed by gradual domain randomization.

## Final Result

Best current checkpoint:

[checkpoints/dagger_hold_dr10_dr0.1_seed42_1777437143.pt](checkpoints/dagger_hold_dr10_dr0.1_seed42_1777437143.pt)

Evaluation command used:

```powershell
python -B evaluate.py `
  --checkpoint checkpoints\dagger_hold_dr10_dr0.1_seed42_1777437143.pt `
  --controller policy `
  --balance-reset-prob 1.0 `
  --num-episodes 100 `
  --label dagger_hold_dr10_policy
```

Main result file:

[results/eval_dagger_hold_dr10_policy.csv](results/eval_dagger_hold_dr10_policy.csv)

Result plot:

[results/eval_dagger_hold_dr10_policy.png](results/eval_dagger_hold_dr10_policy.png)

Summary of the final DR10 policy, evaluated from near-upright starts:

| Fixed mismatch | Mean Max Hold Steps | Hold Score | Balanced Time | Interpretation |
|---:|---:|---:|---:|---|
| -20% | 100.5 | 10.1% | 76.9% | Frequently balanced, but not long continuous holds |
| -15% | 787.6 | 78.8% | 88.4% | Strong |
| -10% | 882.8 | 88.3% | 88.7% | Very strong |
| -5% | 761.1 | 76.1% | 79.7% | Strong |
| -2% | 624.8 | 62.5% | 75.6% | Strong |
| 0% | 496.8 | 49.7% | 71.6% | Good nominal hold |
| +2% | 407.1 | 40.7% | 71.7% | Good near-nominal hold |
| +5% | 273.1 | 27.3% | 66.7% | Partial hold |
| +10% | 49.9 | 5.0% | 19.2% | Weak |
| +15% | 2.8 | 0.3% | 0.3% | Fails |
| +20% | 2.0 | 0.2% | 0.2% | Fails |

The policy is strong at nominal, negative mismatch, and small positive mismatch. It is not yet robust to large positive mismatch. This asymmetry is expected: positive mismatch makes all varied physical parameters heavier/longer/higher inertia at once, reducing effective torque authority.

## Quick Start

Run commands from the repository root:

```powershell
cd "C:\Users\roysw\OneDrive\Documents\GitHub\Hammerhead\TAKE2"
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Evaluate the final checkpoint:

```powershell
python -B evaluate.py `
  --checkpoint checkpoints\dagger_hold_dr10_dr0.1_seed42_1777437143.pt `
  --controller policy `
  --balance-reset-prob 1.0 `
  --num-episodes 100 `
  --label dagger_hold_dr10_policy
```

Render GIFs for representative cases:

```powershell
python -B render_gif.py `
  --checkpoint checkpoints\dagger_hold_dr10_dr0.1_seed42_1777437143.pt `
  --controller policy `
  --balance-reset-prob 1.0 `
  --mismatches -0.10 -0.05 -0.02 0.0 0.02 0.05 0.10 `
  --duration 42 `
  --fps 24 `
  --seed 1
```

Generated GIFs are saved in [results/](results/), for example:

- [results/acrobot_mismatch_minus10pct.gif](results/acrobot_mismatch_minus10pct.gif)
- [results/acrobot_mismatch_plus0pct.gif](results/acrobot_mismatch_plus0pct.gif)
- [results/acrobot_mismatch_plus5pct.gif](results/acrobot_mismatch_plus5pct.gif)
- [results/acrobot_comparison.gif](results/acrobot_comparison.gif)

## Repository Structure

```text
.
|-- Problem statement.pdf          # Assignment brief
|-- train.py                       # PPO loop plus BC/DAgger pretraining
|-- evaluate.py                    # Evaluation across fixed mismatch levels
|-- render_gif.py                  # GIF renderer for trained policies
|-- compare_runs.py                # Helper for comparing checkpoints
|-- requirements.txt               # Python dependencies
|-- envs/
|   |-- randomized_acrobot.py      # Custom Acrobot environment and teacher controller
|   `-- __init__.py
|-- checkpoints/                   # Saved policies
`-- results/                       # Evaluation CSV/PNG and GIF outputs
```

## Problem Definition

The assignment is a robust-control task on Gymnasium Acrobot. The Acrobot is a two-link underactuated pendulum. The controller applies torque at the second joint and must keep both links upright despite physical parameter mismatch.

This implementation focuses on the sustained upright hold objective:

```text
Maximize the longest consecutive streak of balanced steps in a 1000-step episode.
```

Episodes are capped at 1000 steps. The best possible `HoldMax` is therefore 1000.

## Environment

The custom environment is implemented in [envs/randomized_acrobot.py](envs/randomized_acrobot.py).

Nominal parameters:

| Parameter | Nominal value |
|---|---:|
| Link 1 mass | 1.0 kg |
| Link 2 mass | 1.0 kg |
| Link 1 length | 1.0 m |
| Link 2 length | 1.0 m |
| Moment of inertia | 1.0 kg m^2 |
| Link COM position | 0.5 of link length |

Varied parameters during domain randomization:

```text
link_mass_1
link_mass_2
link_length_1
link_length_2
link_moi
```

At each randomized training reset:

```text
p_i ~ Uniform(p_nominal * (1 - dr_range), p_nominal * (1 + dr_range))
```

At evaluation, all varied parameters are shifted by one fixed mismatch value:

```text
p_eval = p_nominal * (1 + mismatch)
```

Evaluation mismatch levels:

```text
-20%, -15%, -10%, -5%, -2%, 0%, +2%, +5%, +10%, +15%, +20%
```

## Observation and Action Space

Observation dimension: 10

The observation contains:

- `cos(theta_1)`, `sin(theta_1)`
- `cos(theta_2)`, `sin(theta_2)`
- `theta_dot_1`, `theta_dot_2`
- `cos(theta_1 + theta_2)`, `sin(theta_1 + theta_2)`
- normalized tip height
- normalized current hold progress

Action dimension: 11

Discrete torque values:

```text
-5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5
```

## Balance Condition

A step is counted as balanced when all of these are true:

- tip height is at least `TARGET_HEIGHT`
- link 1 is within 10 degrees of vertical upright
- link 2 is within 10 degrees of vertical upright
- physical link velocities are at most `1.5 rad/s`
- relative joint velocity is at most `1.5 rad/s`

The fully upright tip height is approximately 2.0 for nominal 1 m + 1 m links. `TARGET_HEIGHT = 1.0` is a height gate; the stricter upright definition comes from the angle and velocity checks.

## Metrics

The primary metric is `Mean Max Hold Steps`.

| Metric | Meaning |
|---|---|
| `HoldMax` / `mean_max_hold_steps` | Average longest consecutive balanced streak |
| `HoldScore` / `hold_score_pct` | `HoldMax / 1000 * 100` |
| `BalTime` / `mean_balance_time_pct` | Percent of episode steps satisfying the full balance condition |
| `Upright` / `mean_upright_time_pct` | Percent of episode steps in the upright/capture region |
| `Return` | Shaped reward sum; useful diagnostically, but not the main success metric |

`Return` can be misleading. Earlier PPO runs achieved high return while `HoldMax` stayed near zero. Final selection is based on hold streak and balance time, not return.

## Training Method

The code supports PPO, behavior cloning, and DAgger-style dataset aggregation.

The successful path was not plain PPO. Direct PPO and teacher-guided PPO initially failed because the policy could collect shaped reward or rely on teacher actions without learning a stable closed-loop balance controller.

The final workflow was:

1. Train a nominal upright hold policy with DAgger-style imitation.
2. Reset data collection when the policy falls out of the capture/upright region.
3. Evaluate policy-only behavior from near-upright starts.
4. Gradually widen domain randomization: nominal -> DR2 -> DR5 -> DR10.
5. Stop before large positive mismatch because the policy still degrades at +10% and above.

### What DAgger Means Here

DAgger stands for Dataset Aggregation.

Simple behavior cloning trains on states visited by the teacher. That was not enough here: the learned policy made small errors, visited states the teacher rarely produced, and failed to recover.

DAgger fixes this by letting the current policy act for part of data collection, while still labeling each visited state with the teacher action:

```text
state comes from current policy rollout
label comes from env.unwrapped.expert_action()
```

The important training flags are:

```text
--pretrain-policy-fraction 0.85
--pretrain-reset-on-fall
```

`--pretrain-policy-fraction` controls how often the current policy, rather than the teacher, executes the rollout action during BC data collection. `--pretrain-reset-on-fall` resets collection when the rollout leaves the capture region, keeping the dataset focused on upright recovery.

## Reproducing the Final Training Sequence

The final checkpoint was produced by incremental DAgger training. The exact checkpoint names include timestamps from the local run, so your filenames will differ.

### 1. Nominal hold pretraining

```powershell
python -B train.py `
  --exp-name dagger_hold_reset_on_fall `
  --total-timesteps 4096 `
  --dr-range 0.0 `
  --balance-reset-prob 1.0 `
  --pretrain-bc-steps 1500 `
  --pretrain-bc-batch-size 256 `
  --pretrain-eval-interval 100 `
  --pretrain-reset-fraction 0.05 `
  --pretrain-policy-fraction 0.70 `
  --pretrain-reset-on-fall `
  --teacher-action-prob 0.0 `
  --teacher-final-prob 0.0 `
  --bc-coef 0.0
```

### 2. Harden nominal hold

```powershell
python -B train.py `
  --resume-checkpoint checkpoints\dagger_hold_reset_on_fall_dr0.0_seed42_1777421072.pt `
  --exp-name dagger_hold_harden `
  --total-timesteps 0 `
  --dr-range 0.0 `
  --balance-reset-prob 1.0 `
  --pretrain-bc-steps 3000 `
  --pretrain-bc-batch-size 256 `
  --pretrain-eval-interval 100 `
  --pretrain-reset-fraction 0.10 `
  --pretrain-policy-fraction 0.85 `
  --pretrain-reset-on-fall `
  --teacher-action-prob 0.0 `
  --teacher-final-prob 0.0 `
  --bc-coef 0.0
```

### 3. Domain randomization at +/-2%

```powershell
python -B train.py `
  --resume-checkpoint checkpoints\dagger_hold_harden_dr0.0_seed42_1777430702.pt `
  --exp-name dagger_hold_dr2 `
  --total-timesteps 0 `
  --dr-range 0.02 `
  --balance-reset-prob 1.0 `
  --pretrain-bc-steps 3000 `
  --pretrain-bc-batch-size 256 `
  --pretrain-eval-interval 100 `
  --pretrain-reset-fraction 0.10 `
  --pretrain-policy-fraction 0.85 `
  --pretrain-reset-on-fall `
  --teacher-action-prob 0.0 `
  --teacher-final-prob 0.0 `
  --bc-coef 0.0
```

### 4. Domain randomization at +/-5%

```powershell
python -B train.py `
  --resume-checkpoint checkpoints\dagger_hold_dr2_dr0.02_seed42_1777432638.pt `
  --exp-name dagger_hold_dr5 `
  --total-timesteps 0 `
  --dr-range 0.05 `
  --balance-reset-prob 1.0 `
  --pretrain-bc-steps 3000 `
  --pretrain-bc-batch-size 256 `
  --pretrain-eval-interval 100 `
  --pretrain-reset-fraction 0.10 `
  --pretrain-policy-fraction 0.85 `
  --pretrain-reset-on-fall `
  --teacher-action-prob 0.0 `
  --teacher-final-prob 0.0 `
  --bc-coef 0.0
```

### 5. Domain randomization at +/-10%

```powershell
python -B train.py `
  --resume-checkpoint checkpoints\dagger_hold_dr5_dr0.05_seed42_1777434529.pt `
  --exp-name dagger_hold_dr10 `
  --total-timesteps 0 `
  --dr-range 0.10 `
  --balance-reset-prob 1.0 `
  --pretrain-bc-steps 3000 `
  --pretrain-bc-batch-size 256 `
  --pretrain-eval-interval 100 `
  --pretrain-reset-fraction 0.10 `
  --pretrain-policy-fraction 0.85 `
  --pretrain-reset-on-fall `
  --teacher-action-prob 0.0 `
  --teacher-final-prob 0.0 `
  --bc-coef 0.0
```

## Results and Artifacts

Latest result artifacts:

| Run | Checkpoint | CSV | Plot |
|---|---|---|---|
| Hardened nominal | [checkpoint](checkpoints/dagger_hold_harden_dr0.0_seed42_1777430702.pt) | [CSV](results/eval_dagger_hold_harden_policy.csv) | [PNG](results/eval_dagger_hold_harden_policy.png) |
| DR2 | [checkpoint](checkpoints/dagger_hold_dr2_dr0.02_seed42_1777432638.pt) | [CSV](results/eval_dagger_hold_dr2_policy.csv) | [PNG](results/eval_dagger_hold_dr2_policy.png) |
| DR5 | [checkpoint](checkpoints/dagger_hold_dr5_dr0.05_seed42_1777434529.pt) | [CSV](results/eval_dagger_hold_dr5_policy.csv) | [PNG](results/eval_dagger_hold_dr5_policy.png) |
| DR10 | [checkpoint](checkpoints/dagger_hold_dr10_dr0.1_seed42_1777437143.pt) | [CSV](results/eval_dagger_hold_dr10_policy.csv) | [PNG](results/eval_dagger_hold_dr10_policy.png) |

Progression of nominal hold:

| Policy | Trained DR range | Nominal HoldMax | Nominal BalTime |
|---|---:|---:|---:|
| Hardened nominal | 0% | 482.8 | 72.7% |
| DR2 | +/-2% | 544.2 | 72.7% |
| DR5 | +/-5% | 575.9 | 73.6% |
| DR10 | +/-10% | 496.8 | 71.6% |

Progression at selected mismatch levels:

| Policy | -10% Hold | -5% Hold | 0% Hold | +5% Hold | +10% Hold |
|---|---:|---:|---:|---:|---:|
| Hardened nominal | 730.5 | 727.8 | 482.8 | 237.0 | 19.2 |
| DR2 | 868.9 | 794.7 | 544.2 | 265.7 | 48.5 |
| DR5 | 896.3 | 807.7 | 575.9 | 299.3 | 60.6 |
| DR10 | 882.8 | 761.1 | 496.8 | 273.1 | 49.9 |

The policy is consistently stronger for negative mismatch than for positive mismatch. The final DR10 policy is robust around nominal and negative mismatch, but large positive mismatch remains the main failure mode.

## Rendering

Render representative GIFs from the final DR10 policy:

```powershell
python -B render_gif.py `
  --checkpoint checkpoints\dagger_hold_dr10_dr0.1_seed42_1777437143.pt `
  --controller policy `
  --balance-reset-prob 1.0 `
  --mismatches -0.10 0.0 0.05 `
  --duration 42 `
  --fps 24 `
  --seed 1
```

GIF outputs:

- [results/acrobot_mismatch_minus10pct.gif](results/acrobot_mismatch_minus10pct.gif)
- [results/acrobot_mismatch_plus0pct.gif](results/acrobot_mismatch_plus0pct.gif)
- [results/acrobot_mismatch_plus5pct.gif](results/acrobot_mismatch_plus5pct.gif)
- [results/acrobot_comparison.gif](results/acrobot_comparison.gif)

Rendering uses one seed at a time. If a GIF reports `Max hold 0`, that single seed produced a failure case. The evaluation CSV is more reliable because it averages across 100 episodes.

## Full Swing-Up Experiment

The validated results above are for robust upright hold from near-upright starts. A full down-start swing-up policy is a separate, harder problem because the controller must first inject energy, then arrive near upright with low enough velocity for the hold controller to stabilize.

An experimental curriculum for this is provided in:

[experiments/full_swingup/README.md](experiments/full_swingup/README.md)

It starts from an older checkpoint that can swing the tip high and then gradually removes near-upright reset assistance. This experiment is intentionally separate from the main reported results so the validated hold-controller artifacts remain unchanged.

## Lessons Learned

1. PPO reward alone was not a reliable success signal. Return increased while `HoldMax` stayed near zero.
2. Teacher-guided PPO could produce balanced rollouts, but the learned policy did not necessarily internalize the behavior.
3. Plain behavior cloning had high action-label accuracy but poor closed-loop stability.
4. DAgger-style data collection with reset-on-fall produced the first useful learned hold controller.
5. Domain randomization had to be widened gradually. Jumping directly to a wide mismatch range made the task harder without improving the core hold behavior.
6. Positive mismatch is harder than negative mismatch, likely because increased mass/length/inertia reduces effective torque authority.

## Requirements

Install with:

```powershell
pip install -r requirements.txt
```

Rendering requires Gymnasium classic-control dependencies, including `pygame`. This is why [requirements.txt](requirements.txt) uses:

```text
gymnasium[classic-control]
pygame
```

## References

- [Problem statement.pdf](Problem%20statement.pdf)
- CleanRL: https://github.com/vwxyzjn/cleanrl
- Gymnasium Acrobot: https://gymnasium.farama.org/environments/classic_control/acrobot/
- Schulman et al., Proximal Policy Optimization Algorithms, 2017
- Ross et al., A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, 2011
- Tobin et al., Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World, 2017
