"""
visualize.py — Generate GIFs of the Acrobot swing-up + balance policy
======================================================================
Creates per-level GIFs and a side-by-side comparison GIF.
Each GIF shows:
  - The acrobot swinging up AND holding the upright position
  - A colour bar showing PHASE: SWING-UP  →  BALANCING  →  SUCCESS / FAILED
  - An upright counter bar filling up toward the 100-step goal

Dependencies:
    pip install pillow imageio pygame

On headless servers (no display) the script sets SDL_VIDEODRIVER=offscreen
automatically — no Xvfb wrapper needed.

Usage
-----
    python visualize.py
    python visualize.py --levels "-0.2,-0.1,0.0,0.1,0.2" --fps 30
    python visualize.py --model-path results/ppo_acrobot.pth --n-tries 5
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont

from common import Agent, AcrobotBalanceEnv, DomainRandomizedAcrobot
from common import UPRIGHT_HEIGHT, BALANCE_STEPS_WIN


def _load_agent(model_path: str, device: str = "cpu") -> Agent:
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    agent = Agent(ckpt["obs_dim"], ckpt["n_actions"]).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent


def _roll_episode(
    agent: Agent,
    mismatch_level: float,
    seed: int = 0,
    device: str = "cpu",
    max_steps: int = 1000,
):
    env = gym.make("Acrobot-v1", render_mode="rgb_array",
                   max_episode_steps=max_steps)
    env = AcrobotBalanceEnv(env)
    env = DomainRandomizedAcrobot(env, randomize=False,
                                  mismatch_level=mismatch_level)
    obs, _ = env.reset(seed=seed)

    frames        = []
    meta          = []   # (tip_height, upright_count) per frame
    total_return  = 0.0
    max_consec    = 0
    cur_consec    = 0
    terminated_   = False

    for _ in range(max_steps):
        frames.append(env.render())
        action = agent.act_single(obs, device=device)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)

        th = AcrobotBalanceEnv.tip_height(obs)
        if th >= UPRIGHT_HEIGHT:
            cur_consec += 1
            max_consec  = max(max_consec, cur_consec)
        else:
            cur_consec  = 0
        meta.append((th, cur_consec))

        if terminated or truncated:
            terminated_ = bool(terminated)
            frames.append(env.render())
            meta.append(meta[-1])
            break

    env.close()
    return frames, meta, total_return, max_consec, terminated_


def _best_episode(agent, mismatch_level, n_tries, device="cpu"):
    best = ([], [], float("-inf"), 0, False)
    for seed in range(n_tries):
        result = _roll_episode(agent, mismatch_level, seed=seed, device=device)
        _, _, ret, consec, _ = result
        if consec > best[3] or (consec == best[3] and ret > best[2]):
            best = result
    return best


def _get_font(size=12):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _annotate_frame(
    frame: np.ndarray,
    mismatch: float,
    tip_h: float,
    cur_consec: int,
    ep_return: float,
    success: bool,
) -> Image.Image:
    img  = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    font_sm = _get_font(11)
    font_md = _get_font(13)

    # Phase detection
    if cur_consec >= BALANCE_STEPS_WIN:
        phase      = "SUCCESS ✓"
        hdr_color  = (39, 174, 96)
    elif tip_h >= UPRIGHT_HEIGHT:
        phase      = f"BALANCING  {cur_consec}/{BALANCE_STEPS_WIN}"
        hdr_color  = (41, 128, 185)
    else:
        phase      = "SWING-UP"
        hdr_color  = (100, 100, 100)

    # Header bar
    header_h = 28
    draw.rectangle([0, 0, W, header_h], fill=hdr_color)
    draw.text((6, 6),   f"Mismatch {mismatch:+.0%}", fill=(255,255,255), font=font_sm)
    draw.text((W//2-40, 6), phase,                    fill=(255,255,255), font=font_md)
    draw.text((W-75, 6), f"ret={ep_return:.0f}",      fill=(255,255,255), font=font_sm)

    # Balance progress bar at bottom
    bar_h  = 10
    bar_y  = H - bar_h
    frac   = min(cur_consec / BALANCE_STEPS_WIN, 1.0)
    draw.rectangle([0, bar_y, W, H], fill=(40, 40, 40))
    bar_color = (39, 174, 96) if frac >= 1.0 else (41, 128, 185)
    draw.rectangle([0, bar_y, int(W * frac), H], fill=bar_color)
    draw.text((4, bar_y), f"Balance: {cur_consec}/{BALANCE_STEPS_WIN}",
              fill=(255,255,255), font=font_sm)
    return img


def _frames_to_gif(frames, meta, path, fps=30,
                   mismatch=0.0, ep_return=0.0, success=False):
    if not frames:
        return
    images = []
    for i, f in enumerate(frames):
        th, cc = meta[i] if i < len(meta) else (0.0, 0)
        img = _annotate_frame(f, mismatch, th, cc, ep_return, success)
        images.append(img.convert("P", dither=Image.Dither.NONE,
                                  palette=Image.Palette.ADAPTIVE))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000/fps), loop=0, optimize=False)


def _side_by_side(all_data: dict, path: str, fps: int = 20):
    """Comparison GIF: one column per mismatch level."""
    levels  = sorted(all_data.keys())
    max_len = max(len(d["frames"]) for d in all_data.values())

    ref_h, ref_w = all_data[levels[0]]["frames"][0].shape[:2]
    label_h = 32
    cell_w, cell_h = ref_w, ref_h + label_h
    total_w = cell_w * len(levels)
    total_h = cell_h
    font = _get_font(12)

    def pad(lst, n, last):
        return lst + [last] * (n - len(lst))

    composite = []
    for t in range(max_len):
        canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        for col, lvl in enumerate(levels):
            d    = all_data[lvl]
            fpad = pad(d["frames"], max_len, d["frames"][-1])
            mpad = pad(d["meta"],   max_len, d["meta"][-1])
            frame = fpad[t]
            th, cc = mpad[t]
            ret    = d["return"]
            succ   = d["success"]

            cell = Image.fromarray(frame).resize((cell_w, ref_h),
                                                 Image.Resampling.NEAREST)
            canvas.paste(cell, (col * cell_w, label_h))

            draw = ImageDraw.Draw(canvas)
            if succ:
                hdr_col = (39, 174, 96)
            elif th >= UPRIGHT_HEIGHT:
                hdr_col = (41, 128, 185)
            else:
                hdr_col = (80, 80, 80)
            draw.rectangle([col*cell_w, 0, (col+1)*cell_w-1, label_h-1], fill=hdr_col)
            lbl = f"{lvl:+.0%}  bal={cc}/{BALANCE_STEPS_WIN}  r={ret:.0f}"
            draw.text((col*cell_w + 3, 8), lbl, fill=(255,255,255), font=font)

        composite.append(canvas.convert("P", dither=Image.Dither.NONE,
                                        palette=Image.Palette.ADAPTIVE))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    composite[0].save(path, save_all=True, append_images=composite[1:],
                      duration=int(1000/fps), loop=0, optimize=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",     type=str, default="./results/ppo_acrobot.pth")
    p.add_argument("--levels",         type=str, default="-0.2,-0.1,0.0,0.1,0.2")
    p.add_argument("--n-tries",        type=int, default=3)
    p.add_argument("--fps",            type=int, default=30)
    p.add_argument("--comparison-fps", type=int, default=20)
    p.add_argument("--gif-dir",        type=str, default="./gifs")
    p.add_argument("--no-comparison",  action="store_true")
    a = p.parse_args()

    levels = [float(x.strip()) for x in a.levels.split(",")]

    if not os.path.exists(a.model_path):
        print(f"\n  ERROR: model not found at '{a.model_path}'")
        print("  Run 'python train.py' first.\n")
        sys.exit(1)

    agent = _load_agent(a.model_path)

    print(f"\n{'='*60}")
    print(f"  Acrobot Swing-Up + Balance Visualizer")
    print(f"{'='*60}")
    print(f"  Levels    : {[f'{l:+.0%}' for l in levels]}")
    print(f"  Tries/lvl : {a.n_tries}  (saves episode with best balance)")
    print(f"  Output    : {a.gif_dir}/")
    print(f"{'='*60}\n")

    os.makedirs(a.gif_dir, exist_ok=True)
    all_data = {}

    for lvl in levels:
        print(f"  Rendering {lvl:+.0%} ...", end=" ", flush=True)
        frames, meta, ret, max_c, succ = _best_episode(
            agent, lvl, n_tries=a.n_tries)

        safe = f"{lvl:+.0f}pct".replace("+", "pos").replace("-", "neg")
        gif_path = os.path.join(a.gif_dir, f"acrobot_balance_{safe}.gif")
        _frames_to_gif(frames, meta, gif_path, fps=a.fps,
                       mismatch=lvl, ep_return=ret, success=succ)

        all_data[lvl] = {"frames": frames, "meta": meta,
                         "return": ret, "success": succ}
        status = f"bal={max_c}/{BALANCE_STEPS_WIN} {'✓ SUCCESS' if succ else '✗ FAILED'}"
        print(f"{len(frames)} frames  |  {status}")
        print(f"    → {gif_path}")

    if not a.no_comparison:
        comp_path = os.path.join(a.gif_dir, "acrobot_comparison.gif")
        print(f"\n  Building comparison GIF ...", end=" ", flush=True)
        _side_by_side(all_data, comp_path, fps=a.comparison_fps)
        print(f"done → {comp_path}")

    print(f"\n  ✓ All GIFs saved to '{a.gif_dir}/'\n")


if __name__ == "__main__":
    main()
