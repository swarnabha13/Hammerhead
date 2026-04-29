# Robust Acrobot Project

## Main Submission

The main project submission is in [TAKE2/](TAKE2/).

Start with the TAKE2 report:

[TAKE2/README.md](TAKE2/README.md)

That README contains the final problem framing, implementation details, training method, evaluation commands, result tables, generated artifacts, and lessons learned. The TAKE2 code and artifacts should be treated as the primary deliverable for this repository.

## Repository Map

| Location | Role |
|---|---|
| [TAKE2/](TAKE2/) | Main project submission and final report |
| [TAKE2/README.md](TAKE2/README.md) | Primary writeup/report |
| [TAKE2/train.py](TAKE2/train.py) | Final training code, including PPO infrastructure and BC/DAgger-style pretraining |
| [TAKE2/evaluate.py](TAKE2/evaluate.py) | Final evaluation script for mismatch sweeps |
| [TAKE2/render_gif.py](TAKE2/render_gif.py) | GIF renderer for the final policy |
| [TAKE2/results/](TAKE2/results/) | Final evaluation plots, CSV files, and GIF artifacts |
| [TAKE2/checkpoints/](TAKE2/checkpoints/) | Saved final and intermediate policies |
| [TAKE3/](TAKE3/) | Later experimental PPO swing-up-and-balance attempt, kept for reference |
| Root-level Python files | Earlier PPO/domain-randomization implementation, kept as support and development history |

## Approaches Tried Before TAKE2

### Root-level implementation

The original project files in the repository root implemented a CleanRL-style PPO agent with domain randomization and dense reward shaping. The goal was to train a single policy that could swing the Acrobot up and keep it upright under simulator parameter mismatch.

Main ideas tried there:

- PPO as the core reinforcement-learning algorithm.
- Dense reward based on tip height, velocity penalty, and upright bonus.
- Domain randomization over link lengths, link masses, and moment of inertia.
- Evaluation across positive, negative, and random parameter mismatch settings.
- Plotting scripts for robustness curves, returns, and heatmaps.

This approach was useful for establishing the project structure and evaluation protocol, but it was not the final reported solution. In practice, PPO reward optimization was not a reliable proxy for sustained upright holding: policies could improve shaped return without learning a stable closed-loop balance behavior.

Relevant root-level files:

- [train.py](train.py)
- [evaluate.py](evaluate.py)
- [visualize.py](visualize.py)
- [run_experiment.py](run_experiment.py)
- [writeup.md](writeup.md)

### TAKE3 implementation

[TAKE3/](TAKE3/) was a separate attempt to solve the harder full swing-up-and-balance task with PPO. It introduced a custom `AcrobotBalanceEnv`, longer episodes, a larger policy network, richer reward shaping, and visual GIF generation.

Main ideas tried there:

- PPO for full down-start swing-up plus sustained balance.
- Suppressing the standard Acrobot termination so the policy had to keep balancing after reaching the upright region.
- A 1000-step episode horizon to allow swing-up, settling, and hold behavior.
- Dense reward terms for height, link posture, velocity control, and sustained-balance bonus.
- Domain randomization across a broader set of physical parameters, including masses, lengths, center-of-mass positions, and moment of inertia.
- GIF visualization for qualitative inspection of swing-up and balance behavior.

TAKE3 is kept as a reference experiment, but it is not the final submission. The final TAKE2 direction narrowed the validated task to robust upright hold from near-upright starts and used DAgger-style imitation from a stabilizing teacher, followed by gradual domain randomization. That produced the strongest documented results.

Relevant TAKE3 files:

- [TAKE3/README.md](TAKE3/README.md)
- [TAKE3/train.py](TAKE3/train.py)
- [TAKE3/evaluate.py](TAKE3/evaluate.py)
- [TAKE3/visualize.py](TAKE3/visualize.py)
- [TAKE3/run_all.py](TAKE3/run_all.py)

## Final Direction

The final TAKE2 solution is based on the lesson that direct PPO reward maximization was not enough for stable holding. TAKE2 keeps the useful infrastructure from the earlier attempts, but shifts the successful training path to:

1. Learn a near-upright hold controller from a stabilizing teacher.
2. Use DAgger-style data aggregation so the policy sees states caused by its own mistakes.
3. Reset data collection after falls to focus learning on the upright capture region.
4. Gradually expand domain randomization from nominal dynamics to wider mismatch ranges.
5. Evaluate the learned policy across fixed mismatch levels and report hold-time metrics.

For the complete report, commands, results, and references, see [TAKE2/README.md](TAKE2/README.md).
