# Robust Acrobot Balance - PPO + Domain Randomization

> **Take-Home Project**: Train a policy that can balance the Acrobot upright despite
> mismatch between training and evaluation dynamics.

---

## Quick Start

Run commands from the repository root:

```powershell
cd "C:\Users\roysw\OneDrive\Documents\GitHub\Hammerhead\TAKE2"
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the domain-randomized policy:

```powershell
python train.py `
  --exp-name ppo_robust_dr5_hold `
  --dr-range 0.05 `
  --total-timesteps 10000000 `
  --num-envs 16 `
  --seed 42
```

Evaluate the newest checkpoint across mismatch levels:

```powershell
$ckpt = Get-ChildItem checkpoints\ppo_robust_dr5_hold*.pt |
  Where-Object { $_.Name -ne "latest.pt" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

python evaluate.py `
  --checkpoint $ckpt.FullName `
  --num-episodes 100 `
  --label "PPO_DR5_hold" `
  --out-dir results
```

Compare all saved checkpoints, if you trained multiple runs:

```powershell
python compare_runs.py --auto --num-episodes 100 --out-dir results
```

Generate GIFs for nominal and +/-2% mismatch:

```powershell
python render_gif.py `
  --checkpoint $ckpt.FullName `
  --mismatches -0.02 0.0 0.02 `
  --duration 42 `
  --fps 24 `
  --out-dir results
```

The evaluation outputs are saved to `results/` as CSV/PNG files. GIFs are saved to
`results/acrobot_mismatch_*.gif` and `results/acrobot_comparison.gif`.

**Quick smoke test** (checks code paths, not final performance):

```powershell
python train.py `
  --exp-name smoke_hold `
  --dr-range 0.05 `
  --total-timesteps 2048 `
  --num-envs 2 `
  --num-steps 64 `
  --num-minibatches 2 `
  --update-epochs 1 `
  --save-dir checkpoints\smoke
```

You can also run the scripted pipeline:

```bash
bash run_experiment.sh
```

---

## Project Structure

```
robust_acrobot/
├── train.py                   # PPO training loop (CleanRL-based)
├── evaluate.py                # Systematic evaluation across mismatch levels
├── compare_runs.py            # Side-by-side comparison of multiple policies
├── render_gif.py              # Animated GIF renderer for trained policies
├── run_experiment.sh          # Full pipeline script
├── requirements.txt           # Python dependencies
├── envs/
│   ├── __init__.py
│   └── randomized_acrobot.py  # Custom Acrobot with parameter injection
├── checkpoints/               # Saved model checkpoints (created at runtime)
└── results/                   # CSV + PNG evaluation results (created at runtime)
```

---

## 1. Approach and Design Choices

### Algorithm: PPO (Proximal Policy Optimization)

**Why PPO?**

PPO was selected as the base algorithm from CleanRL for the following reasons:

| Criterion | Reasoning |
|---|---|
| **Stability** | Clipped surrogate objective prevents destructive policy updates |
| **Sample efficiency** | Multiple gradient epochs per rollout (reuses data safely) |
| **Discrete actions** | Acrobot uses 5 bounded torque actions; PPO with Categorical policy is natural |
| **Community baseline** | Extensive ablation studies available; bugs are well-known |
| **CleanRL compatibility** | Direct implementation from CleanRL reference |

DQN was considered but rejected: off-policy methods are more sensitive to distribution shift
when the environment dynamics change between training and evaluation (exactly our setting).
PPO's on-policy nature makes the interaction between domain randomization and data collection
more predictable.

**Modifications from base CleanRL PPO:**
- Hidden size increased to 256 (from 64) to accommodate two-link balancing and varied dynamics
- Shared trunk architecture (actor + critic share a feature extractor) for efficiency
- `fixed_mismatch` evaluation mode added to the environment
- Domain randomization applied at environment level, transparent to the PPO algorithm

### Network Architecture

```
Observation (10,)
      │
  Linear(10 -> 256) + Tanh      <- Shared representation
  Linear(256 -> 256) + Tanh     <- Shared representation
      │
  ┌───┴───────────────────┐
  │                       │
Actor Head             Critic Head
Linear(256 -> 11)       Linear(256 -> 1)
  │                       │
Categorical π(a|s)     V(s) ∈ ℝ
```

**Why Tanh activations?** Tanh keeps activations bounded, which aids training stability when
the observation magnitudes change under domain randomization (e.g., faster dynamics produce
larger velocity observations). ReLU can suffer from "dead neurons" in this setting.

---

## 2. Sim-to-Real Pipeline & Parameters

### Environment: Acrobot-v1

The Acrobot is a 2-link underactuated pendulum. The agent controls a bounded torque on the
second joint (11 discrete levels from `-5` to `+5` N·m), must swing the free end
above the target line, and then hold both links near vertical upright.

**State space** (10-dimensional):
- cos(θ₁), sin(θ₁) — angle of link 1 from downward vertical
- cos(θ₂), sin(θ₂) — angle of link 2 relative to link 1
- θ̇₁, θ̇₂ — angular velocities
- cos(θ₁ + θ₂), sin(θ₁ + θ₂) — absolute angle of the outer link
- Normalized free-end height and hold-progress fraction

**Reward**: shaped reward based on both links being upright, end-effector height, velocity
damping, and a bonus for remaining in the upright balance region.

### Scope Assumptions

This implementation treats simulator-to-real mismatch as **physical parameter uncertainty only**.
It does not model control latency, sensor noise, actuator friction, backlash, observation delay,
or other unmodeled system dynamics.

Torque limits are handled by a bounded discrete action space: the policy chooses one of 11
torques from `-5` to `+5` N·m. A small torque penalty discourages excessive
actuation, but no extra penalty is added for jerk or switching frequency. The task objective
is to reach and hold both links upright within the episode time limit.

### Physical Parameters Varied

| Parameter | Symbol | Nominal | DR Range (training) | Rationale |
|---|---|---|---|---|
| Link 1 mass | m₁ | 1.0 kg | ±5% | Dominant inertia term |
| Link 2 mass | m₂ | 1.0 kg | ±5% | Affects centripetal forces |
| Link 1 length | l₁ | 1.0 m | ±5% | Changes moment arm |
| Link 2 length | l₂ | 1.0 m | ±5% | Changes torque leverage |
| Moment of inertia | I | 1.0 kg·m² | ±5% | Directly scales rotational dynamics |

**Parameters held fixed**: Link COM positions (0.5 — halfway along each link). These are
geometrically well-defined and less likely to drift in a real system.

### Why These Parameters?

These five parameters collectively define the **equations of motion** of the Acrobot.
Together they determine:
1. How quickly the links respond to applied torque (inertia terms)
2. How strong gravity's effect is (mass × length terms)
3. The coupling dynamics between links (cross-inertia terms)

A policy that is robust to variation in these parameters must learn fundamentally general
swing-up strategies rather than memorising the exact resonant frequency of the nominal system.

### Why ±5% Training Range?

Domain randomization theory suggests training range should **exceed** the expected evaluation
mismatch. The primary balance target is nominal and ±2% mismatch, so ±5% training randomization
covers the target range while keeping the harder sustained two-link balance task learnable.
We still evaluate wider mismatch levels to show where the controller degrades.

Training with a wider range such as ±10% or ±20% is possible, but it makes the balance objective
substantially harder and should be treated as a follow-up once nominal and ±2% are reliable.

---

## 3. Robustness Strategy: Domain Randomization (DR)

**Domain Randomization** trains the policy on a *distribution* of MDPs rather than a single
fixed environment. At each episode reset, parameters are sampled independently:

```
pᵢ ~ Uniform(pᵢ_nominal × (1 − δ), pᵢ_nominal × (1 + δ))    δ = 0.10
```

The trained policy must succeed across all sampled parameter combinations. Since the evaluation
mismatch levels are drawn from the same type of distribution (uniform deviations from nominal),
a policy that solves the DR training distribution should transfer to fixed mismatches.

**Why DR over other robustness methods?**

| Method | Pros | Cons | Verdict |
|---|---|---|---|
| **Domain Randomization** ✓ | Simple, no extra components, well-studied | May be conservative | **Chosen** |
| Robust Adversarial RL | Optimises for worst-case | Computationally expensive, can be too conservative | Nice-to-have |
| System Identification | Adapts to specific env | Requires real-world data | Not applicable here |
| MAML/Meta-RL | Fast adaptation | Complex, high compute | Nice-to-have |

DR is the minimum viable robustness technique: it requires no changes to the PPO algorithm,
adds negligible compute overhead, and has strong empirical support in sim-to-real literature.

---

## 4. Balance Objective

### Definition

> **Objective**: maximize the longest consecutive streak that the Acrobot keeps both links
> near upright during the 1000-step episode.
>
> Episodes run until the time limit. There is no fixed 500-step termination threshold.

The balance region is defined as:
- Link 1 absolute angle within `10 degrees` of vertical upright
- Link 2 absolute angle within `10 degrees` of vertical upright
- Absolute angular velocity of each physical link, plus relative joint velocity, at most `1.5 rad/s`

The reward has two explicit phases:
- **Swing-up phase**: active before the robot reaches the capture region. The reward prioritizes lifting and moving the links toward upright.
- **Capture/balance phase**: active once the robot is high or near upright. The reward shifts toward stabilizing both links, damping velocity, and extending the current hold streak.

### Metrics Reported

| Metric | Formula | Why chosen |
|---|---|---|
| **Mean Max Hold Steps** | Mean longest consecutive near-upright streak | Directly measures the objective we optimize |
| **Hold Score %** | Mean max hold steps / 1000 | Normalized version of the hold streak |
| **Mean Return** | Mean cumulative shaped reward | Captures both swing-up quality and upright control |
| **Upright Time %** | % episode steps where the outer link is within 10 degrees of vertical | Measures phase-2 entry reliability |
| **Balanced Time %** | % episode steps satisfying the full balance condition | Measures sustained robust balance quality |

**Primary metric: Mean Max Hold Steps.** A policy that keeps the Acrobot balanced for
longer consecutive streaks is better, even if it does not hit an arbitrary fixed threshold.
Return and time-in-region metrics provide secondary quality signals.

### Justification

The Acrobot balance task is better represented as a sustained-control objective than as a
binary solved/not-solved event. Measuring the longest consecutive balanced streak gives
credit for partial improvement while still penalizing policies that merely swing through the
upright region.


---

## 5. Evaluation Protocol

**Test conditions**: DR is disabled during evaluation. Parameters are fixed at:
```
p_eval = p_nominal × (1 + mismatch_level)    for each varied parameter
```

**Mismatch levels tested**: −20%, −15%, −10%, −5%, −2%, 0%, +2%, +5%, +10%, +15%, +20%

Note: negative mismatch (lighter/shorter links) tends to change the natural frequency differently
than positive mismatch, so we test both directions. Results often show asymmetry — see Analysis.

**Protocol**:
1. Load frozen policy (no gradient updates)
2. Run 100 episodes at each mismatch level with fixed random seed
3. Record: longest hold streak, hold score, episode return, and episode length
4. Aggregate metrics across episodes

---

## 6. Expected Results

Based on domain randomization theory, we expect:

| Mismatch Level | Expected Max Hold | Notes |
|---|---|---|
| 0% (nominal) | High hold streak | Near-optimal on training distribution center |
| ±2% | High hold streak | Well within training DR range |
| ±5% | High hold streak | Comfortably within DR range |
| ±10% | Moderate-to-high hold streak | At the DR boundary — some degradation |
| ±15% | Moderate hold streak | Slightly outside DR range |
| ±20% | Lower but nonzero hold streak | Outside DR range — graceful degradation expected |

**Baseline (no DR)**: Expected to hold well near nominal but degrade faster at larger mismatch levels.

---

## 7. Results Overview

After switching from the default Acrobot threshold task to the sustained-balance objective,
the policy must be retrained before reporting final numbers. Follow the Quick Start commands
above to train, evaluate, compare runs, and generate GIFs.

The generated results will be saved in `results/` as CSV files, plots, and GIFs. Older result
files from the threshold-crossing task should not be interpreted as hold-streak balance results.

---

## 8. Analysis

### Why DR succeeds within the training range (±5%)

During training, the policy must solve Acrobot for every parameter combination in the DR
distribution. This forces it to learn a general swing-up strategy: build momentum by
exploiting the natural pendulum dynamics, then redirect energy to reach the goal. This
strategy works across a range of inertias because it relies on the *structure* of the task
rather than memorising exact timings.

### Why performance may degrade at ±20%

At +20% mass (heavier links), the applied torque becomes insufficient to generate rapid swing-up.
The agent may need more steps and its strategy (learned for up to +10% heavier links) may not
generalise fully. At -20% (lighter links), the system becomes more responsive but also more
sensitive to timing errors. Both extremes expose edge cases not seen during training.

### Asymmetry: positive vs. negative mismatch

Physical intuition:
- **Positive mismatch** (heavier/longer): more inertia → harder to swing up → more steps needed
- **Negative mismatch** (lighter/shorter): less inertia → faster dynamics → timing becomes critical

A policy trained in the nominal regime often handles negative mismatch better because the
system just becomes "easier" (lower inertia), while positive mismatch imposes a fundamental
capability limit (less torque authority relative to load).

### Potential improvements

1. **Wider DR range** (±10-20% training): would improve wider evaluation mismatch at cost of slower training
2. **Robust RL (minimax)**: explicitly optimise for worst-case parameter → no degradation
3. **Privileged information**: feed current physics parameters to the policy during training
   (not available at eval) — allows the policy to adapt at runtime
4. **Parameter estimation**: add a system-identification head that infers parameters from rollout
   data, then condition the policy on estimated parameters

---

## 9. Hyperparameter Reference

| Hyperparameter | Value | Source |
|---|---|---|
| Total timesteps | 10,000,000 | More rollout data for sustained two-link balance |
| Num envs | 16 | Parallel data collection |
| Steps per rollout | 256 | Longer rollouts for swing-up plus balance |
| Learning rate | 2.5×10⁻⁴ | CleanRL PPO default |
| LR annealing | Linear decay to 0 | Standard PPO trick |
| γ (discount) | 0.99 | Standard |
| λ (GAE) | 0.95 | Standard |
| Clip coefficient | 0.2 | Standard PPO |
| Entropy coefficient | 0.01 | Encourages exploration |
| Value function coef | 0.5 | Standard |
| Update epochs | 10 | Reuses each batch 10× |
| Minibatches | 4 | Batch size / 4 |
| Max grad norm | 0.5 | Prevents gradient explosion |

---

## 10. References

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017). arXiv:1707.06347
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks* (2017). arXiv:1703.06907
- CleanRL: https://github.com/vwxyzjn/cleanrl
- Gymnasium Acrobot: https://gymnasium.farama.org/environments/classic_control/acrobot/
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2018), Chapter 11 (Off-policy)
- Lilian Weng, *Domain Randomization for Sim2Real Transfer* (2019). https://lilianweng.github.io

---

## 11. Citation

If you use this code, please cite:

```bibtex
@misc{robust_acrobot_2024,
  title  = {Robust Acrobot Balance via PPO and Domain Randomization},
  year   = {2024},
  note   = {Take-home RL project},
}
```
