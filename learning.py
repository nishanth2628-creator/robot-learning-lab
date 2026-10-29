"""Finite-horizon tabular Q-learning on a discrete, fixed obstacle map.

No physics, sensors or robot hardware are modelled. Coordinates are (x, y).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left
ACTION_NAMES = ("up", "right", "down", "left")
DEFAULT_HORIZON = 50
TRAIN_SLIP = 0.10


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    obstacles: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]

    def __post_init__(self):
        if not isinstance(self.width, int) or not isinstance(self.height, int) or min(self.width, self.height) < 1:
            raise ValueError("Grid dimensions must be positive integers")
        if any(not self.inside(cell) for cell in self.obstacles):
            raise ValueError("Obstacle outside map")
        if not self.free(self.start) or not self.free(self.goal):
            raise ValueError("Start and goal must be free cells")

    def inside(self, cell):
        return (len(cell) == 2 and all(isinstance(v, (int, np.integer)) for v in cell)
                and 0 <= cell[0] < self.width and 0 <= cell[1] < self.height)

    def free(self, cell):
        return self.inside(cell) and cell not in self.obstacles


def demo_grid():
    return Grid(9, 7, frozenset({(3, 0), (3, 1), (3, 3), (3, 4), (4, 4),
                               (5, 4), (7, 4), (5, 1), (5, 2)}), (0, 0), (8, 6))


@dataclass(frozen=True)
class Observation:
    position: tuple[int, int]
    remaining: int


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    collision: bool
    actual_action: int

    @property
    def done(self):
        return self.terminated or self.truncated


class Environment:
    """A collision means a blocked move attempt, not entering an obstacle.

    An action slips to its left with p/2, right with p/2, or executes as
    commanded with 1-p. Exactly one environment random draw is used per step.
    """
    def __init__(self, grid, slip=TRAIN_SLIP, horizon=DEFAULT_HORIZON):
        if not 0 <= slip <= 1 or not math.isfinite(slip):
            raise ValueError("Slip must be a finite probability")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Horizon must be a positive integer")
        self.grid, self.slip, self.horizon = grid, slip, horizon
        self.rng = None

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.position, self.remaining = self.grid.start, self.horizon
        self.done = self.grid.start == self.grid.goal
        return Observation(self.position, self.remaining)

    def step(self, action):
        if self.rng is None or self.done:
            raise RuntimeError("Reset before stepping; completed episodes cannot continue")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError("Action must be an integer in [0, 3]")
        noise = self.rng.random()
        actual = ((action - 1) % 4 if noise < self.slip / 2 else
                  (action + 1) % 4 if noise < self.slip else action)
        dx, dy = ACTIONS[actual]
        proposed = self.position[0] + dx, self.position[1] + dy
        collision = not self.grid.free(proposed)
        if not collision:
            self.position = proposed
        self.remaining -= 1
        terminated = self.position == self.grid.goal
        truncated = self.remaining == 0 and not terminated
        # Terminal goal reward replaces the step cost. Blocked attempts cost -5.
        reward = 20.0 if terminated else (-5.0 if collision else -1.0)
        self.done = terminated or truncated
        return Transition(Observation(self.position, self.remaining), reward,
                          terminated, truncated, collision, actual)


def shortest_path(grid, start=None):
    """Shortest four-connected deterministic path, including both endpoints.

    Returns None for disconnected free endpoints. It ignores slip and rewards.
    """
    start = grid.start if start is None else start
    if not grid.free(start):
        raise ValueError("Start must be a free grid cell")
    frontier, parents = deque([start]), {start: None}
    while frontier:
        current = frontier.popleft()
        if current == grid.goal:
            path = [current]
            while parents[path[-1]] is not None:
                path.append(parents[path[-1]])
            return list(reversed(path))
        for dx, dy in ACTIONS:
            nxt = current[0] + dx, current[1] + dy
            if grid.free(nxt) and nxt not in parents:
                parents[nxt] = current
                frontier.append(nxt)
    return None


def bfs_action(grid, position):
    """Replan a deterministic shortest path at the actual observed cell."""
    path = shortest_path(grid, position)
    if path is None or len(path) < 2:
        raise ValueError("BFS policy needs a reachable goal distinct from its position")
    delta = path[1][0] - position[0], path[1][1] - position[1]
    return ACTIONS.index(delta)


def q_target(reward, next_max, gamma, terminated=False, truncated=False):
    """No bootstrapping on either true termination or the finite-horizon limit."""
    return float(reward if terminated or truncated else reward + gamma * next_max)


def state_values(q_table, observation):
    x, y = observation.position
    return q_table[observation.remaining, y, x]


def train(grid, seed=7, episodes=12000, horizon=DEFAULT_HORIZON,
          slip=TRAIN_SLIP, alpha=0.25, gamma=0.97, log_every=100):
    if episodes < 1 or not 0 < alpha <= 1 or not 0 <= gamma <= 1 or log_every < 1:
        raise ValueError("Invalid training configuration")
    # Remaining time is part of the state: this is a finite-horizon MDP.
    q_table = np.zeros((horizon + 1, grid.height, grid.width, 4), dtype=np.float64)
    rng = np.random.default_rng(seed)
    env = Environment(grid, slip, horizon)
    history, block = [], []
    for episode in range(episodes):
        # These seeds are separate from all declared evaluation seeds.
        env_seed = 20_000_000 + seed * 1_000_000 + episode
        obs = env.reset(env_seed)
        epsilon = max(0.05, 0.90 * (1 - episode / max(1, episodes * 0.80)))
        total_reward, collisions, steps = 0.0, 0, 0
        while not env.done:
            values = state_values(q_table, obs)
            if rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                # Random ties in training avoid a hard-coded directional bias.
                action = int(rng.choice(np.flatnonzero(values == values.max())))
            transition = env.step(action)
            target = q_target(transition.reward, state_values(q_table, transition.observation).max(),
                              gamma, transition.terminated, transition.truncated)
            values[action] += alpha * (target - values[action])
            total_reward += transition.reward
            collisions += int(transition.collision)
            steps += 1
            obs = transition.observation
        block.append((total_reward, env.position == grid.goal, collisions, steps))
        if (episode + 1) % log_every == 0 or episode + 1 == episodes:
            array = np.asarray(block, dtype=float)
            history.append({"seed": seed, "episode": episode + 1, "episodes_in_block": len(block),
                            "epsilon_at_block_end": epsilon, "mean_return": float(array[:, 0].mean()),
                            "success_rate": float(array[:, 1].mean()),
                            "mean_collisions": float(array[:, 2].mean()), "mean_steps": float(array[:, 3].mean())})
            block = []
    return q_table, history


@dataclass
class Rollout:
    seed: int
    positions: list
    actions: list
    rewards: list
    collisions: list
    success: bool
    timeout: bool


def rollout(grid, policy, seed, q_table=None, slip=TRAIN_SLIP, horizon=DEFAULT_HORIZON):
    if policy not in {"q_learning", "random", "bfs"}:
        raise ValueError("Unknown policy")
    if policy == "q_learning" and (q_table is None or q_table.shape != (horizon+1, grid.height, grid.width, 4)):
        raise ValueError("Q table must match grid and horizon")
    env = Environment(grid, slip, horizon)
    obs = env.reset(seed)
    # Random policy decisions have their own stream, independent of environment noise.
    policy_rng = np.random.default_rng(2_000_000_000 + seed)
    positions, actions, rewards, collisions = [obs.position], [], [], []
    while not env.done:
        if policy == "q_learning":
            # Evaluation has no epsilon/exploration and deterministic argmax ties.
            action = int(state_values(q_table, obs).argmax())
        elif policy == "random":
            action = int(policy_rng.integers(4))
        else:
            action = bfs_action(grid, obs.position)
        transition = env.step(action)
        positions.append(transition.observation.position)
        actions.append(action)
        rewards.append(transition.reward)
        collisions.append(transition.collision)
        obs = transition.observation
    return Rollout(seed, positions, actions, rewards, collisions,
                   env.position == grid.goal, env.remaining == 0 and env.position != grid.goal)


def wilson_interval(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z / count
    centre = (p + z*z/(2*count)) / denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0.0, centre-half), min(1.0, centre+half)]


def evaluate(grid, policy, seeds, q_table=None, slip=TRAIN_SLIP, horizon=DEFAULT_HORIZON):
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use a nonempty collection of distinct evaluation seeds")
    before = None if q_table is None else q_table.copy()
    view = None if q_table is None else q_table.view()
    if view is not None:
        view.flags.writeable = False
    trials = [rollout(grid, policy, seed, view, slip, horizon) for seed in seeds]
    if before is not None and not np.array_equal(before, q_table):
        raise AssertionError("Evaluation modified the Q table")
    successes = sum(t.success for t in trials)
    per_episode = [{"seed": t.seed, "success": t.success, "timeout": t.timeout,
                    "steps": len(t.actions), "return": sum(t.rewards),
                    "blocked_move_attempts": sum(t.collisions)} for t in trials]
    successful_steps = [len(t.actions) for t in trials if t.success]
    return {"episodes": len(trials), "successes": successes,
            "success_rate": successes/len(trials), "success_rate_wilson_95": wilson_interval(successes, len(trials)),
            "mean_return": float(np.mean([sum(t.rewards) for t in trials])),
            "mean_steps_all_episodes": float(np.mean([len(t.actions) for t in trials])),
            "mean_steps_successes_only": float(np.mean(successful_steps)) if successful_steps else None,
            "mean_blocked_move_attempts": float(np.mean([sum(t.collisions) for t in trials])),
            "timeouts": sum(t.timeout for t in trials), "evaluation_updates": 0,
            "q_unchanged": before is None or np.array_equal(before, q_table),
            "per_episode": per_episode}
