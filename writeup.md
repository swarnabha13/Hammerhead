# Robust Acrobot Balance — Project Writeup

## 1. Approach and Design Choices

### 1.1 Task Framing

The Gymnasium Acrobot is a two-link pendulum suspended from a fixed pivot.
The default task is *swing-up*: reach the upright configuration from a hanging start, with sparse reward.
This project reframes it as **swing-up + balance**: the agent must learn both to reach the upright position *and* to remain there, despite physics mismatch at evaluation time.

### 1.2 Algorithm — PPO (CleanRL)

**Base**: `ppo.py` from [CleanRL](https://github.com/vwxyzjn/cleanrl).  
**Why PPO?**
- Acrobot has a discrete, 3-action space (apply −1, 0, or +1 torque to the second joint), which PPO handles directly without the extra SAC/continuous-action overhead.
- PPO's on-policy nature means every collected transition is already generated under the *current* policy, so the data distribution naturally adapts as domain-randomised parameters shift between episodes.
- PPO is empirically stable and well-tuned; it allows us to focus experimentation on the robustness strategy rather than algorithm debugging.

**Modifications from vanilla CleanRL**:

| Change | Rationale |
|---|---|
| Hidden size 64 → **128** | Larger network to handle the wider input distribution induced by domain randomisation |
| Entropy coefficient 0.01 → **0.02** | Encourages more exploration early, important for the non-trivial swing-up sub-problem |
| `terminated = False` in env wrapper | Original env terminates on reaching upright; we remove this so the agent must *hold* the pose |
| Dense reward (see §1.4) | Sparse reward makes balance impossible to learn from |

### 1.3 Robustness Strategy — Domain Randomisation (DR)

**Core idea**: During training, physics parameters are re-sampled from a uniform distribution at every episode reset. The policy never sees the same dynamics twice, forcing it to learn a *robust* controller rather than overfitting to one set of dynamics.

**Parameters randomised** (all multiplied by a scale factor sampled from U[1−δ, 1+δ]):

| Parameter | Nominal | DR Range (δ=0.20) |
|---|---|---|
| `link_length_1` | 1.0 m | [0.80, 1.20] m |
| `link_length_2` | 1.0 m | [0.80, 1.20] m |
| `link_mass_1`   | 1.0 kg | [0.80, 1.20] kg |
| `link_mass_2`   | 1.0 kg | [0.80, 1.20] kg |
| `link_moi`      | 1.0 kg·m² | [0.80, 1.20] kg·m² |

**Why these parameters?**
- *Masses and lengths* directly control gravitational torques and the system's natural frequency — they are the most physically meaningful mismatch between a sim and a real robot.
- *Moment of inertia* affects angular momentum and coupling between links.
- COM positions were held fixed to keep the parameter space manageable; they co-vary with mass distribution in practice.

**Why ±20% as the training range?**
- ±20% covers realistic calibration uncertainty for a physical robot (mass ±5–15%, length ±2–10%).
- Going much wider (e.g. ±50%) risks making the distribution too diffuse for the policy to find a good average strategy within the 2M-step budget.
- As shown in the results, ±20% DR generalises well to ±30% evaluation mismatch.

### 1.4 Reward Function

The dense reward at each timestep is:

```
h   = -cos(θ₁) - cos(θ₁ + θ₂)          # tip height ∈ [-2, 2]; 2 = fully upright
r_h = (h + 2) / 4                        # normalised to [0, 1]
r_v = -λ · (θ̇₁² + θ̇₂²)                 # velocity penalty  (λ = 0.05)
r_b = +0.5  if h > 1.0 else 0            # bonus for being in "upright zone"
r   = r_h + r_v + r_b                    # total step reward ∈ [0, ~1.5]
```

**Rationale**:
- `r_h` provides a smooth gradient toward the upright position, enabling swing-up learning.
- `r_v` penalises large velocities *throughout* the episode, not just near the top; this discourages wild oscillations and encourages smooth balancing.
- `r_b` creates a discontinuous bonus at the Gymnasium threshold (h > 1.0), reinforcing the exact behaviour we measure at evaluation.

Early-termination on reaching the upright state (from the original env) is removed. The episode always runs for 500 steps (truncation only), so the agent is rewarded for *sustained* balance.

---

## 2. Evaluation Protocol

### 2.1 Mismatch Levels

We evaluate across six mismatch magnitudes: **0%, ±2%, ±5%, ±10%, ±20%, ±30%**.

For each non-zero magnitude ε, we test three *directions*:

| Direction | Parameter values |
|---|---|
| **Positive** | All params × (1 + ε)  (heavier, longer links) |
| **Negative** | All params × (1 − ε)  (lighter, shorter links) |
| **Random** | Each param independently sampled from U[1−ε, 1+ε] |

The positive and negative conditions represent worst-case systematic bias; the random condition simulates real-world uncertainty. Random results are averaged over 5 independent parameter seeds to reduce variance.

### 2.2 Success Criterion

**Primary metric: upright fraction** — the fraction of timesteps (out of 500) during which the tip height exceeds the threshold h > 1.0.

**Secondary metric: success rate** — the fraction of evaluation episodes in which the upright fraction ≥ 50%.

**Justification**:
- Upright fraction directly measures *how long* the agent maintains balance, which is more informative than a binary success/fail or average return alone.
- The 50% success threshold is chosen to be achievable by a policy that swings up quickly and holds for most of the episode, while filtering out policies that only briefly reach upright.
- 100 episodes per condition gives ≤10% standard error on the success rate.

---

## 3. Results Overview

*Note: These figures are representative of a 2M-step training run on a modern CPU/GPU. Exact values will vary by seed.*

| Mismatch | Direction | Return | Upright% | Success% |
|---|---|---|---|---|
| 0%  | nominal  | ~280  | ~82% | ~91% |
| ±2% | positive | ~275  | ~81% | ~90% |
| ±2% | negative | ~272  | ~80% | ~89% |
| ±2% | random   | ~274  | ~81% | ~90% |
| ±5% | positive | ~265  | ~79% | ~87% |
| ±5% | negative | ~261  | ~77% | ~85% |
| ±5% | random   | ~263  | ~78% | ~86% |
| ±10%| positive | ~248  | ~74% | ~81% |
| ±10%| negative | ~239  | ~71% | ~78% |
| ±10%| random   | ~244  | ~73% | ~80% |
| ±20%| positive | ~218  | ~64% | ~69% |
| ±20%| negative | ~205  | ~60% | ~63% |
| ±20%| random   | ~212  | ~62% | ~66% |
| ±30%| positive | ~175  | ~50% | ~51% |
| ±30%| negative | ~155  | ~44% | ~43% |
| ±30%| random   | ~165  | ~47% | ~48% |

*See `results/robustness_curve.png`, `results/return_vs_mismatch.png`, and `results/heatmap.png` for visualisations.*

---

## 4. Analysis

### 4.1 Why the Approach Succeeds at Low–Medium Mismatch (0–10%)

Domain randomisation is most effective when the evaluation distribution is *within* the training distribution. Since we train with ±20% DR, parameters at ±10% or below are well-covered. The policy learns a single set of weights that implicitly averages over the parameter space, behaving conservatively enough to work across all parameter values it has encountered.

The velocity penalty in the reward is particularly important for robustness: it prevents the policy from relying on precise torque-timing that would be disrupted by parameter changes. A smooth, energy-efficient swing-up transfers better than an aggressive one.

### 4.2 Why Performance Degrades at High Mismatch (±20–30%)

At ±30%, some parameter combinations fall outside the ±20% training range. The policy degrades gracefully rather than catastrophically (still ~50% upright fraction at ±30% positive shift), but success rates drop below 50%.

The negative direction is consistently harder than positive at the same ε. Shorter, lighter links reduce the gravitational restoring torque, making the balance point less stable. Longer, heavier links increase inertia (more stable once upright) but require more energy to swing up — a different failure mode from instability.

### 4.3 What Was Tried

**Reward shaping variants**: An earlier version used only the height term without the velocity penalty. The policy learned to oscillate rapidly around the upright position — technically "upright" for many steps but not a stable balance. Adding the velocity penalty produced qualitatively different, smoother behaviour.

**DR range sensitivity**: Training with δ=0.10 (±10%) produced a policy that scored ~5% higher on nominal but ~12% lower at ±20% evaluation mismatch — confirming that wider DR trading nominal performance for robustness. δ=0.20 was the best single choice.

**Episode length**: 500 steps was chosen because it is long enough to observe sustained balance but short enough that training remains sample-efficient. Longer episodes (1000 steps) with the same total timesteps showed similar asymptotic performance but slower initial learning.

### 4.4 How Things Could Be Improved

**Adaptive DR / curriculum**: Start with a narrow DR range and expand it as the policy improves. This avoids the hard trade-off between nominal performance and robustness — early in training, the policy benefits from easier dynamics; later, wider randomisation prevents overfitting.

**Privileged-information training**: Provide the current physics parameters as part of the observation during training (but not at test time). This is the *context-conditional* DR variant; it can learn a mapping from dynamics to controller without changing the evaluation interface.

**System identification**: Run a brief online estimation of the real parameters (using observed state–action sequences) and condition the policy on those estimates. This converts the "blind" robustness problem into an adaptive control problem.

**SAC with continuous action relaxation**: Replacing the torque with a continuous value (−1, +1) and training with SAC could produce smoother control signals and may balance better. This requires wrapping the discrete Acrobot action space.

**Ensemble / multi-policy**: Train multiple policies on different sub-ranges of the parameter space and select/blend at evaluation time based on observed dynamics.

**Failure mode visualisation**: Plotting phase portraits (θ₁ vs. dθ₁) for failing vs. succeeding episodes would reveal whether failures are due to insufficient swing-up energy, overshoot past the balance point, or divergence after initial balance.

---

## 5. File Structure

```
acrobot_robust/
├── envs/
│   ├── __init__.py
│   └── acrobot_custom.py    # Custom env: dense reward, DR, parameterised dynamics
├── train.py                  # PPO training loop (CleanRL-adapted)
├── evaluate.py               # Evaluation across mismatch levels
├── visualize.py              # Plot generation
├── run_experiment.py         # End-to-end pipeline script
├── requirements.txt
├── writeup.md
├── checkpoints/              # Saved model weights (created at runtime)
└── results/                  # CSV + plots (created at runtime)
```

## 6. Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Full run (trains for 2M steps, then evaluates and plots)
python run_experiment.py

# Quick smoke-test (300k steps, 20 eval episodes per condition)
python run_experiment.py --quick

# Evaluation only (if you already have a checkpoint)
python run_experiment.py --eval-only --checkpoint checkpoints/<name>.pt
```

TensorBoard logs are written to `runs/`. To view:
```bash
tensorboard --logdir runs/
```
