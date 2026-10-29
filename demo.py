"""Train three seeded policies, compare frozen evaluation, and export evidence."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import platform
import time
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from learning import (DEFAULT_HORIZON, TRAIN_SLIP, demo_grid, evaluate, rollout,
                      shortest_path, train)

TRAINING_SEEDS = (7, 19, 43)
EVALUATION_SEEDS = tuple(range(100000, 100100))
SCENARIOS = {"no_slip": 0.0, "nominal": 0.10, "shifted_slip": 0.30, "stress_slip": 0.50}


class Figure:
    """Produce the same small figure in portable SVG and PNG."""
    def __init__(self, width=1600, height=1260):
        self.image = Image.new("RGB", (width, height), "#f7f9fc")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<title>Robot Learning Lab: measured synthetic navigation experiment</title>',
                    '<desc>Q-learning, BFS replanning and random policies on one fixed grid, with held-out action noise.</desc>',
                    f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>']
        self.fonts = {}

    def rectangle(self, box, fill, outline=None):
        x1, y1, x2, y2 = box
        self.draw.rectangle(box, fill=fill, outline=outline)
        self.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="{fill}" stroke="{outline or "none"}"/>')

    def line(self, points, color, width=2):
        if len(points) < 2:
            return
        self.draw.line(points, fill=color, width=width, joint="curve")
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')

    def circle(self, center, radius, color):
        x, y = center
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>')

    def text(self, x, y, value, size=20, color="#15273b", bold=False):
        key = size, bold
        if key not in self.fonts:
            for name in ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
                         "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
                try:
                    self.fonts[key] = ImageFont.truetype(name, size)
                    break
                except OSError:
                    pass
            else:
                self.fonts[key] = ImageFont.load_default(size=size)
        self.draw.text((x, y), str(value), font=self.fonts[key], fill=color, anchor="lt")
        self.svg.append(f'<text x="{x}" y="{y+size*0.86}" fill="{color}" font-size="{size}" font-family="Arial,sans-serif" font-weight="{700 if bold else 400}">{escape(str(value))}</text>')

    def save(self, directory):
        self.image.save(directory / "demo.png")
        (directory / "demo.svg").write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")


def make_figure(grid, q_table, history, measurements, directory):
    fig = Figure()
    muted, teal, orange = "#586d83", "#007e83", "#c66a27"
    fig.text(64, 34, "Robot Learning Lab", 42, bold=True)
    fig.text(64, 92, "Tabular Q-learning  |  finite horizon  |  stochastic action slip", 24, muted)
    fig.line([(64, 141), (1536, 141)], "#dbe3ed", 2)
    fig.text(64, 176, "One fixed map: first held-out episode", 26, bold=True)
    fig.text(64, 213, "10% slip; environment seed 100000", 20, muted)
    cell, left, top = 52, 64, 259
    for y in range(grid.height):
        for x in range(grid.width):
            color = "#334b60" if (x, y) in grid.obstacles else "#ffffff"
            fig.rectangle((left+x*cell, top+y*cell, left+(x+1)*cell, top+(y+1)*cell), color, "#dbe3ed")

    def point(position):
        return left+(position[0]+0.5)*cell, top+(position[1]+0.5)*cell

    # Both are actual rollouts with the same episode noise, not cherry-picked paths.
    for policy, color, width in (("bfs", orange, 8), ("q_learning", teal, 4)):
        trial = rollout(grid, policy, 100000, q_table, slip=TRAIN_SLIP)
        fig.line([point(p) for p in trial.positions], color, width)
    for pos, label in ((grid.start, "S"), (grid.goal, "G")):
        px, py = point(pos)
        fig.circle((px, py), 14, "#15273b")
        fig.text(px-7, py-11, label, 20, "#ffffff", bold=True)
    fig.line([(64, 655), (102, 655)], teal, 4)
    fig.text(115, 643, "Frozen greedy Q", 20, muted)
    fig.line([(310, 655), (348, 655)], orange, 8)
    fig.text(361, 643, "BFS replan", 20, muted)
    fig.text(64, 690, "A blocked move stays in its current cell.", 19, muted)
    fig.text(64, 717, "No sensors, physics or robot hardware.", 19, muted)

    fig.text(650, 176, "Learning from interaction", 26, bold=True)
    fig.text(650, 213, "Exploratory training return: means of 100 episodes", 20, muted)
    chart_x, chart_y, chart_w, chart_h = 710, 272, 790, 300
    values = [row["mean_return"] for row in history]
    low = min(-120, int(min(values) // 20) * 20)
    high = 20
    max_episode = max(row["episode"] for row in history)
    for value in range(low, high+1, 20):
        py = chart_y + chart_h - (value-low)/(high-low)*chart_h
        fig.line([(chart_x, py), (chart_x+chart_w, py)], "#dbe3ed", 1)
        fig.text(chart_x-55, py-10, value, 17, muted)
    colors = (teal, "#cf892b", "#7463b5")
    for seed, color in zip(TRAINING_SEEDS, colors):
        rows = [r for r in history if r["seed"] == seed]
        points = [(chart_x+r["episode"]/max_episode*chart_w,
                   chart_y+chart_h-(r["mean_return"]-low)/(high-low)*chart_h) for r in rows]
        fig.line(points, color, 3)
    for x, label in ((chart_x, "0"), (chart_x+chart_w/2-30, str(max_episode//2)),
                     (chart_x+chart_w-65, str(max_episode))):
        fig.text(x, chart_y+chart_h+15, label, 18, muted)
    fig.text(chart_x+300, chart_y+chart_h+45, "training episodes", 18, muted)
    for index, (seed, color) in enumerate(zip(TRAINING_SEEDS, colors)):
        x = 710+index*245
        fig.line([(x, 666), (x+34, 666)], color, 3)
        fig.text(x+45, 654, f"training seed {seed}", 18, muted)
    fig.text(650, 702, "The primary model is seed 7, chosen in advance.", 20, muted)
    fig.text(650, 731, "No evaluation episode updates Q or uses exploration.", 20, muted)

    fig.text(64, 789, "Measured evaluation: 100 held-out noise seeds per policy and condition", 25, bold=True)
    xs = (80, 430, 635, 870, 1120, 1375)
    titles = ("Slip / policy", "Goal rate", "Mean return", "Blocked moves", "Mean steps", "Timeouts")
    fig.rectangle((64, 835, 1536, 881), "#e6edf5")
    for x, title in zip(xs, titles):
        fig.text(x, 848, title, 19, bold=True)
    row_index = 0
    for scenario in ("nominal", "shifted_slip", "stress_slip"):
        for policy, name in (("q_learning", "Q-learning"), ("bfs", "BFS replan"), ("random", "Random")):
            row = measurements["evaluation"][scenario][policy]
            y = 884+row_index*30
            if row_index // 3 % 2 == 0:
                fig.rectangle((64, y-1, 1536, y+29), "#ffffff")
            label = f"{int(SCENARIOS[scenario]*100)}% / {name}"
            items = (label, f"{row['successes']}/100", f"{row['mean_return']:.2f}",
                     f"{row['mean_blocked_move_attempts']:.2f}", f"{row['mean_steps_all_episodes']:.2f}", str(row["timeouts"]))
            for x, value in zip(xs, items):
                fig.text(x, y+5, value, 19, teal if policy == "q_learning" else "#15273b")
            row_index += 1
    fig.text(64, 1182, "Fixed-map experiment only. BFS is shortest without slip; it is not an optimal noisy-policy benchmark.", 21, muted)
    fig.text(64, 1215, "AI-assisted educational portfolio  |  Full results, uncertainty and training-seed variation: metrics.json", 20, muted)
    fig.save(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12000, help="Training episodes for each of three seeds")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes >= 1_000_000:
        parser.error("--episodes must be between 1 and 999999")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grid = demo_grid()
    models, history = {}, []
    for seed in TRAINING_SEEDS:
        q, rows = train(grid, seed=seed, episodes=args.episodes)
        models[seed] = q
        history.extend(rows)
    q_table = models[7]
    shortest = shortest_path(grid)
    measurements = {
        "scope": "Same fixed grid in training and evaluation; only episode noise seeds are held out. No unseen-map, physics, ROS or hardware claim.",
        "environment": {"width": grid.width, "height": grid.height,
                        "start": grid.start, "goal": grid.goal, "obstacles": sorted(grid.obstacles),
                        "horizon": DEFAULT_HORIZON, "training_slip_probability": TRAIN_SLIP,
                        "slip_semantics": "Command executes with 1-p; left/right perpendicular action each with p/2.",
                        "reward_semantics": "Goal +20 replaces step cost; other free moves -1; blocked moves -5 and remain in place.",
                        "goal_on_last_step": "termination, not truncation", "shortest_path_no_slip_steps": len(shortest)-1},
        "training": {"algorithm": "finite-horizon tabular Q-learning", "state": "remaining_steps, y, x",
                     "episodes_per_seed": args.episodes, "seeds": list(TRAINING_SEEDS), "primary_seed": 7,
                     "primary_selection": "Declared before evaluation; not best-seed selection.",
                     "alpha": 0.25, "gamma": 0.97, "epsilon_start": 0.90, "epsilon_floor": 0.05,
                     "epsilon_decay": "linear to floor during first 80% of episodes",
                     "q_initialization": "all zeros", "q_shape": list(q_table.shape),
                     "primary_nonzero_q_entries": int(np.count_nonzero(q_table)),
                     "environment_seed_rule": "20,000,000 + training_seed*1,000,000 + episode_index"},
        "evaluation_design": {"episodes_per_condition": len(EVALUATION_SEEDS), "seeds": list(EVALUATION_SEEDS),
                              "epsilon": 0.0, "updates": 0, "q_immutable_view_and_snapshot_check": True,
                              "common_random_numbers": "Every policy/condition receives the same per-episode uniform noise stream; random policy uses a separate decision stream.",
                              "bfs_semantics": "Replans from actual cell after every move; shortest on deterministic grid, ignores slip and reward.",
                              "reported_return": "Undiscounted reward sum; training uses gamma-discounted targets.",
                              "intervals": "95% Wilson intervals describe 100 episode successes for a fixed policy, not training uncertainty."},
        "evaluation": {}, "training_seed_variation": {},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__}}
    for scenario, slip in SCENARIOS.items():
        measurements["evaluation"][scenario] = {
            policy: evaluate(grid, policy, EVALUATION_SEEDS,
                             q_table if policy == "q_learning" else None, slip=slip)
            for policy in ("q_learning", "bfs", "random")}
        measurements["training_seed_variation"][scenario] = {
            str(seed): evaluate(grid, "q_learning", EVALUATION_SEEDS, model, slip=slip)
            for seed, model in models.items()}
    measurements["runtime"]["total_seconds"] = round(time.perf_counter()-started, 3)
    np.savez_compressed(directory / "q_table.npz", q_table=q_table, horizon=DEFAULT_HORIZON,
                        width=grid.width, height=grid.height, start=grid.start, goal=grid.goal,
                        obstacles=np.asarray(sorted(grid.obstacles)), training_seed=7, schema_version=1)
    with (directory / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (directory / "metrics.json").write_text(json.dumps(measurements, indent=2)+"\n", encoding="utf-8")
    make_figure(grid, q_table, history, measurements, directory)
    for scenario in SCENARIOS:
        print(scenario, {p: {k: r[k] for k in ("success_rate", "mean_return", "mean_steps_all_episodes", "mean_blocked_move_attempts")}
                         for p, r in measurements["evaluation"][scenario].items()})
    print("Q seed variation:", {scenario: {seed: (row["success_rate"], row["mean_return"])
                                        for seed, row in values.items()}
                                for scenario, values in measurements["training_seed_variation"].items()})
    print(f"Saved experiment outputs in {directory.resolve()}")


if __name__ == "__main__":
    main()
