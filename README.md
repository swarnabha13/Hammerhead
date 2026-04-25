# Robust Acrobot Balance

Train a PPO policy that can balance the Acrobot upright despite simulator-to-real
physics mismatch, using **Domain Randomisation** as the robustness strategy.

---

## Quick Setup

```bash
# 1. Create and activate a virtual environment (recommended)
conda create --name hammerhead python=3.11
conda activate hammerhead

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full experiment
python run_experiment.py        # trains 2M steps, evaluates, plots (~20-40 min)

# OR: smoke-test in ~3-5 minutes
python run_experiment.py --quick
```

---

## What Each File Does

| File | Purpose |
|---|---|
| `envs/acrobot_custom.py` | Custom Acrobot env: dense reward, no early stop, domain randomisation |
| `train.py` | PPO training loop (CleanRL-adapted) |
| `evaluate.py` | Evaluates frozen policy across 6 mismatch levels × 3 directions |
| `visualize.py` | Generates 3 result plots |
| `run_experiment.py` | **Start here** — orchestrates train → eval → plot |
| `writeup.md` | Full approach, design choices, analysis |

---

## Running Stages Individually

```bash
# Training only
python train.py --total-timesteps 2000000 --dr-range 0.20 --num-envs 8

# Evaluation only (needs a checkpoint)
python evaluate.py --checkpoint checkpoints/NAME_best.pt --n-episodes 100

# Plots only (needs results/eval_results.csv)
python visualize.py
```

---

## Key Design Choices

| Choice | Decision | Reason |
|---|---|---|
| Algorithm | PPO (CleanRL) | Stable, discrete actions, on-policy data adapts naturally to DR |
| Robustness | Domain Randomisation ±20% | No test-time sys-ID needed; covers realistic calibration error |
| Reward | Dense height + velocity penalty + upright bonus | Enables swing-up AND sustained balance |
| Success metric | Upright fraction ≥ 50% over 500-step episode | Measures sustained balance, not just momentary contact |

---

## Parameters Randomised

At every episode reset, these are scaled by U[0.8, 1.2]:

- `link_length_1`, `link_length_2`
- `link_mass_1`, `link_mass_2`
- `link_moi`

---

## Evaluation Protocol

6 mismatch levels: **0%, ±2%, ±5%, ±10%, ±20%, ±30%**

3 directions per level:
- **Positive**: all params × (1 + ε)
- **Negative**: all params × (1 − ε)
- **Random**: each param independently sampled from U[1−ε, 1+ε]

---

## Outputs

After running the experiment:

```
results/
  eval_results.csv          ← raw numbers for all conditions
  robustness_curve.png      ← upright fraction & success rate vs mismatch
  return_vs_mismatch.png    ← episodic return with error bars
  heatmap.png               ← all conditions at a glance

runs/                       ← TensorBoard logs
checkpoints/                ← saved model weights
```

Monitor training live:
```bash
tensorboard --logdir runs/
```

---

## Expected Results (2M-step run)

| Mismatch | Upright % | Success Rate |
|---|---|---|
| 0% nominal | ~80–85% | ~90% |
| ±5% | ~75–80% | ~85% |
| ±10% | ~70–75% | ~78% |
| ±20% | ~60–65% | ~65% |
| ±30% | ~45–55% | ~48% |

See `writeup.md` for full analysis.
