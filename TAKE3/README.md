# Robust Acrobot — Swing-Up AND Balance (PPO + Domain Randomisation)

A sim-to-real robustness pipeline for **swinging the Acrobot upright AND
holding it balanced** with both links vertical, built on top of
[CleanRL PPO](https://github.com/vwxyzjn/cleanrl).

---

## Quick Start

```bash
pip install -r requirements.txt

python run_all.py                        # full pipeline (2M steps)
python run_all.py --generate-gifs        # also render GIFs (needs pygame)

# Or step by step:
python train.py    --total-timesteps 2000000
python evaluate.py --model-path results/ppo_acrobot.pth
python visualize.py
```

Output files in `./results/` and `./gifs/`:

| File | Contents |
|---|---|
| `results/ppo_acrobot.pth` | Frozen policy checkpoint |
| `results/training_curve.png` | Return + balance-success rate over training |
| `results/eval_plot.png` | 4-panel robustness chart |
| `results/eval_results.json` | Per-level metrics |
| `gifs/acrobot_balance_0pct.gif` | Nominal policy (swing-up + hold) |
| `gifs/acrobot_comparison.gif` | Side-by-side comparison across mismatch levels |

---

## What Changed vs. Swing-Up Only

### Problem: Why the standard Acrobot fails at balance

Standard `Acrobot-v1` terminates **immediately** the moment the tip crosses
the line. The agent never has to hold the position — it just needs to cross
once. This produces policies that swing past the line and let the arm fall.

### Fix 1 — `AcrobotBalanceEnv` (custom wrapper)

| Aspect | Standard Acrobot-v1 | AcrobotBalanceEnv |
|---|---|---|
| Termination on tip crossing | ✅ episode ends | ❌ suppressed |
| Success condition | tip_height > 1.0 once | hold tip_height ≥ 1.8 for 100 consecutive steps |
| Reward | −1 per step | Dense shaped (see below) |
| Max episode length | 500 steps | **1000 steps** (more time to swing + settle) |

### Fix 2 — Dense Shaped Reward

```
r_height  = (tip_height + 2) / 4          ∈ [0, 1]  (higher = better)
r_link1   = (−cos θ₁ + 1) / 2             ∈ [0, 1]  (link 1 pointing up)
r_link2   = ( cos θ₂ + 1) / 2             ∈ [0, 1]  (link 2 straight)

r_posture = 0.5·r_height + 0.3·r_link1 + 0.2·r_link2
r_vel     = (r_height²) · 0.05 · (ω₁² + ω₂²)   ← heavier near upright

reward    = r_posture − r_vel
          + 0.5  (bonus per step when tip_height ≥ 1.8)
          + 50.0 (terminal bonus for 100-step sustained balance)
```

The velocity penalty **increases quadratically near the top** — the agent
is rewarded not just for reaching the upright region, but for *slowing down*
once it arrives there.

### Fix 3 — Updated Hyperparameters

| Param | Swing-up only | Swing-up + Balance |
|---|---|---|
| `total_timesteps` | 500 k | **2 M** |
| `num_envs` | 4 | **8** |
| `num_steps` | 128 | **256** |
| `gamma` | 0.99 | **0.995** |
| `ent_coef` | 0.01 | **0.02** |
| Network | 64 × 64 | **128 × 128** |

---

## Approach & Design Choices

### Algorithm — PPO (CleanRL)

PPO with clipped surrogate objective; see CleanRL docs. Chosen for stability
with shaped rewards and discrete action spaces.

### Robustness — Domain Randomisation (DR)

All 7 physics parameters re-sampled on every episode reset during training:

| Parameter | Nominal | Training range (±20%) |
|---|---|---|
| `LINK_LENGTH_1` | 1.0 m | [0.80, 1.20] |
| `LINK_LENGTH_2` | 1.0 m | [0.80, 1.20] |
| `LINK_MASS_1` | 1.0 kg | [0.80, 1.20] |
| `LINK_MASS_2` | 1.0 kg | [0.80, 1.20] |
| `LINK_COM_POS_1` | 0.5 m | [0.40, 0.60] |
| `LINK_COM_POS_2` | 0.5 m | [0.40, 0.60] |
| `LINK_MOI` | 1.0 kg·m² | [0.80, 1.20] |

Training with ±20% covers all evaluation mismatch levels (±2, 5, 10, 20%).

---

## Evaluation Protocol

### Mismatch levels tested

```
−20%  −10%  −5%  −2%  0% (nominal)  +2%  +5%  +10%  +20%
```

### Success criterion

> **An episode is "successful" if the policy holds the upright position
> (tip_height ≥ 1.8) for ≥ 100 consecutive steps before the 1000-step limit.**

This directly measures the **balance** objective, not just tip clearance.

### Four reported metrics

| Metric | What it measures |
|---|---|
| **Success rate** | % episodes with ≥100 consecutive upright steps |
| **Mean balance steps** | Avg max consecutive steps in upright region |
| **Upright fraction** | % of all steps spent upright (tip_height ≥ 1.8) |
| **Mean return** | Average shaped return per episode |

---

## Expected Results

| Mismatch | Success Rate | Balance Steps | Notes |
|---|---|---|---|
| ±2% – ±5% | ~85–95% | 100+ | Comfortably solved |
| ±10% | ~75–90% | 80–100 | Usually solved |
| ±20% | ~60–80% | 60–100 | Edge of training envelope |
| 0% (nominal) | ~88–96% | 100+ | Baseline |

---

## GIF Visualizer

```bash
pip install pillow imageio pygame   # one-time install

python visualize.py                 # 5 mismatch levels + comparison GIF
python visualize.py --levels "-0.2,-0.1,0.0,0.1,0.2" --fps 30 --n-tries 5
```

Each GIF shows:
- 🔘 **SWING-UP phase** — grey header; agent pumping energy
- 🔵 **BALANCING phase** — blue header; blue progress bar filling toward 100 steps
- 🟢 **SUCCESS** — green header; full progress bar
- 🔴 **FAILED** — ran out of steps before 100-step balance achieved

---

## File Structure

```
acrobot-robust-rl/
├── common.py        # AcrobotBalanceEnv, DomainRandomizedAcrobot, Agent (128×128)
├── train.py         # PPO training loop (CleanRL-style, balance hyperparams)
├── evaluate.py      # Mismatch sweep + 4-panel matplotlib chart
├── visualize.py     # GIF generator with phase overlay + progress bar
├── run_all.py       # One-shot: train → evaluate → (optionally) GIFs
├── requirements.txt
└── README.md
```
