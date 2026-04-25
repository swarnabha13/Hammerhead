#!/usr/bin/env bash
# =============================================================================
#  run_experiment.sh  –  Full Robust Acrobot Experiment Pipeline
# =============================================================================
#
#  Runs:
#    1. Training with Domain Randomization (DR ±10%)
#    2. Training a no-DR baseline (for comparison)
#    3. Evaluation of both policies across mismatch levels
#    4. Comparison plots
#
#  Usage:
#    bash run_experiment.sh          # full experiment
#    bash run_experiment.sh quick    # quick run (fewer timesteps, for testing)
# =============================================================================

set -e  # exit on error

MODE=${1:-"full"}

if [ "$MODE" = "quick" ]; then
    STEPS=200000
    ENVS=2
    EPISODES=30
    echo ">>> QUICK MODE (200k steps, 2 envs, 30 eval episodes)"
else
    STEPS=1000000
    ENVS=4
    EPISODES=100
    echo ">>> FULL MODE (1M steps, 4 envs, 100 eval episodes)"
fi

echo "============================================================"
echo "  STEP 1/4 – Install dependencies"
echo "============================================================"
pip install -q -r requirements.txt

echo ""
echo "============================================================"
echo "  STEP 2/4 – Train DR policy (±10% domain randomization)"
echo "============================================================"
python train.py \
    --exp-name ppo_robust_dr10 \
    --dr-range 0.10 \
    --total-timesteps $STEPS \
    --num-envs $ENVS \
    --seed 42

echo ""
echo "============================================================"
echo "  STEP 3/4 – Train no-DR baseline (for comparison)"
echo "============================================================"
python train.py \
    --exp-name ppo_baseline_noDR \
    --dr-range 0.0 \
    --total-timesteps $STEPS \
    --num-envs $ENVS \
    --seed 42

echo ""
echo "============================================================"
echo "  STEP 4/4 – Evaluate both policies"
echo "============================================================"
# Evaluate DR policy
python evaluate.py \
    --checkpoint checkpoints/latest.pt \
    --num-episodes $EPISODES \
    --label "PPO+DR_10pct" \
    --out-dir results

# Evaluate baseline (find it by name pattern)
BASELINE=$(ls -t checkpoints/ppo_baseline_noDR*.pt 2>/dev/null | grep -v latest | head -1)
if [ -n "$BASELINE" ]; then
    python evaluate.py \
        --checkpoint "$BASELINE" \
        --num-episodes $EPISODES \
        --label "PPO_no_DR" \
        --out-dir results
fi

# Comparison plot
python compare_runs.py \
    --auto \
    --num-episodes $EPISODES \
    --out-dir results

echo ""
echo "============================================================"
echo "  DONE! Results saved in results/"
echo "  Checkpoints in checkpoints/"
echo "============================================================"
