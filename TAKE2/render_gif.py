"""
render_gif.py  –  Render a Trained Acrobot Policy as an Animated GIF
======================================================================
Loads a frozen PPO checkpoint and records one (or more) balance episodes
at each requested mismatch level, then stitches the frames into a GIF.

Dependencies (in addition to requirements.txt):
    pip install imageio pillow

Usage
-----
# Render nominal policy (no mismatch)
python render_gif.py

# Render at the target robustness levels
python render_gif.py --mismatches -0.02 0.0 0.02

# Render a specific checkpoint
python render_gif.py --checkpoint checkpoints/my_run.pt

# Control speed and resolution
python render_gif.py --fps 30 --duration 8

Output
------
results/acrobot_mismatch_<level>.gif  for each mismatch level
results/acrobot_comparison.gif        side-by-side strip of all levels
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
from envs.randomized_acrobot import (
    MAX_EPISODE_STEPS,
    BALANCE_HOLD_STEPS,
    OBSERVATION_DIM,
    RandomizedAcrobotEnv,
    TORQUE_VALUES,
)
from train import ActorCritic


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Render a trained Acrobot policy as an animated GIF"
    )
    p.add_argument("--checkpoint",  type=str,   default="checkpoints/latest.pt",
                   help="Path to trained .pt checkpoint")
    p.add_argument("--mismatches",  type=float, nargs="+",
                   default=[-0.02, 0.0, 0.02],
                   help="Mismatch levels to render (e.g. -0.10 0.0 0.10)")
    p.add_argument("--fps",         type=int,   default=24,
                   help="Frames per second in the output GIF")
    p.add_argument("--duration",    type=float, default=21.0,
                   help="Maximum episode duration in seconds")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out-dir",     type=str,   default="results")
    p.add_argument("--no-comparison", action="store_true", default=False,
                   help="Skip the side-by-side comparison GIF")
    p.add_argument("--label-font-size", type=int, default=14)
    p.add_argument("--controller", choices=["policy", "hybrid", "mpc"],
                   default="hybrid",
                   help="Action source: learned policy, MPC expert, or hybrid policy+MPC stabilizer")
    return p.parse_args()


# ─────────────────────────────────────────────
# GIF helpers
# ─────────────────────────────────────────────

def load_agent(ckpt_path: str) -> ActorCritic:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first to generate a checkpoint."
        )
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_obs_dim = ckpt.get("obs_dim", OBSERVATION_DIM)
    if ckpt_obs_dim != OBSERVATION_DIM:
        raise ValueError(
            f"Checkpoint obs_dim={ckpt_obs_dim}, but the current two-link "
            f"balance environment expects obs_dim={OBSERVATION_DIM}. "
            "Retrain the policy with the current code before rendering."
        )
    expected_action_dim = len(TORQUE_VALUES)
    ckpt_action_dim = ckpt.get("action_dim", expected_action_dim)
    if ckpt_action_dim != expected_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={ckpt_action_dim}, but the current two-link "
            f"balance environment expects action_dim={expected_action_dim}. "
            "Retrain the policy with the current code before rendering."
        )
    agent = ActorCritic(OBSERVATION_DIM, expected_action_dim)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


def add_label(frame: np.ndarray, text: str, font_size: int = 14) -> np.ndarray:
    """
    Overlay a text label onto a raw RGB frame (numpy HxWx3 uint8).
    Uses PIL so no display/X11 is required.
    """
    from PIL import Image, ImageDraw, ImageFont

    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to load a decent font; fall back to default bitmap font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Draw drop-shadow + white text at top-left
    shadow_offset = 1
    draw.text((11, 11), text, font=font, fill=(0, 0, 0))        # shadow
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))  # foreground
    return np.array(img)


def render_episode(
    agent:      ActorCritic,
    mismatch:   float,
    max_steps:  int,
    seed:       int,
    font_size:  int,
    controller: str,
) -> list[np.ndarray]:
    """
    Render one episode and return a list of annotated RGB frames.
    Uses 'rgb_array' render mode – no display needed.
    """
    env = RandomizedAcrobotEnv(
        render_mode    = "rgb_array",
        fixed_mismatch = mismatch,
        dr_range       = 0.0,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    obs, _       = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    step         = 0
    success      = False
    last_info: dict = {"target_reached": False, "balance_steps": 0}

    for step in range(max_steps):
        frame = env.render()
        if frame is None:
            break

        # Build status label
        label = (f"Mismatch: {mismatch*100:+.0f}%  |  "
                 f"Step: {step:3d}  |  "
                 f"Return: {total_reward:.0f}  |  "
                 f"Target: {'Y' if last_info.get('target_reached', False) else 'N'}  |  "
                 f"Hold: {int(last_info.get('balance_steps', 0)):02d}/{BALANCE_HOLD_STEPS}")
        if success:
            label += "  BALANCED"

        frames.append(add_label(frame, label, font_size))

        with torch.no_grad():
            if controller == "mpc":
                action_item = env.unwrapped.expert_action()
            elif controller == "hybrid" and env.unwrapped.should_use_balance_controller():
                action_item = env.unwrapped.expert_action()
            else:
                action_item = int(agent.get_deterministic_action(
                    torch.FloatTensor(obs).unsqueeze(0)
                ).item())
        obs, reward, terminated, truncated, last_info = env.step(action_item)
        total_reward += reward

        if terminated:
            success = True
            # Freeze the final balanced frame for 1 second so it's visible
            final_frame = env.render()
            if final_frame is not None:
                final_label = (f"Mismatch: {mismatch*100:+.0f}%  |  "
                               f"Step: {step:3d}  |  Return: {total_reward:.0f}  BALANCED!")
                for _ in range(int(env.metadata.get("render_fps", 50) * 0.8)):
                    frames.append(add_label(final_frame, final_label, font_size))
            break

        if truncated:
            break

    env.close()
    print(f"  Mismatch {mismatch*100:+5.1f}%  |  "
          f"{'BALANCED' if success else 'TIMEOUT':8s}  |  "
          f"{step+1} steps  |  return {total_reward:.0f}  |  "
          f"{len(frames)} frames")
    return frames


def frames_to_gif(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Save a list of RGB numpy frames as an animated GIF using imageio."""
    import imageio.v3 as iio
    duration_ms = int(1000 / fps)
    iio.imwrite(path, frames, extension=".gif",
                plugin="pillow",
                loop=0,
                duration=duration_ms)


def make_comparison_gif(
    all_frames: dict[float, list[np.ndarray]],
    path:       str,
    fps:        int,
) -> None:
    """
    Build a single side-by-side GIF showing all mismatch levels simultaneously.
    All episode streams are padded/truncated to the same length, then tiled
    horizontally into one wide frame.
    """
    from PIL import Image

    if not all_frames:
        return

    levels  = sorted(all_frames.keys())
    streams = [all_frames[k] for k in levels]

    # Unify frame counts: pad shorter episodes by repeating their last frame
    max_len = max(len(s) for s in streams)
    padded  = []
    for stream in streams:
        if len(stream) == 0:
            continue
        pad_needed = max_len - len(stream)
        padded.append(stream + [stream[-1]] * pad_needed)

    if not padded:
        return

    # Tile frames side-by-side
    combined: list[np.ndarray] = []
    for frame_idx in range(max_len):
        row_frames = [Image.fromarray(s[frame_idx]) for s in padded]
        w = sum(f.width  for f in row_frames)
        h = max(f.height for f in row_frames)
        canvas = Image.new("RGB", (w, h), (20, 20, 20))
        x = 0
        for f in row_frames:
            canvas.paste(f, (x, 0))
            x += f.width
        combined.append(np.array(canvas))

    frames_to_gif(combined, path, fps)
    print(f"\n  Comparison GIF → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Check imageio is available
    try:
        import imageio.v3  # noqa: F401
    except ImportError:
        print("ERROR: imageio is required for GIF export.")
        print("Install with:  pip install imageio pillow")
        sys.exit(1)

    # Check PIL is available
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("ERROR: Pillow is required for text rendering.")
        print("Install with:  pip install pillow")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Acrobot Policy Renderer")
    print(f"{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Controller : {args.controller}")
    print(f"  Mismatches : {[f'{m*100:+.0f}%' for m in args.mismatches]}")
    print(f"  FPS        : {args.fps}")
    print(f"  Max dur    : {args.duration}s  ({int(args.duration * args.fps)} max frames)")
    print(f"{'='*60}\n")

    agent    = load_agent(args.checkpoint)
    max_steps = int(args.duration * args.fps)

    all_frames: dict[float, list[np.ndarray]] = {}

    for mismatch in args.mismatches:
        frames = render_episode(
            agent     = agent,
            mismatch  = mismatch,
            max_steps = max_steps,
            seed      = args.seed,
            font_size = args.label_font_size,
            controller= args.controller,
        )
        all_frames[mismatch] = frames

        # Save individual GIF
        level_str = f"{mismatch*100:+.0f}".replace("+", "plus").replace("-", "minus")
        gif_path  = os.path.join(args.out_dir, f"acrobot_mismatch_{level_str}pct.gif")
        frames_to_gif(frames, gif_path, args.fps)
        print(f"  Saved: {gif_path}")

    # Side-by-side comparison GIF
    if not args.no_comparison:
        comp_path = os.path.join(args.out_dir, "acrobot_comparison.gif")
        make_comparison_gif(all_frames, comp_path, fps=min(args.fps, 20))

    print(f"\n  Done! All GIFs saved to {args.out_dir}/")
    print(f"\n  View with any browser or image viewer that supports animated GIF.")


if __name__ == "__main__":
    main()
