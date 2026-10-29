"""Meaningful environment, learning-target and evaluation-isolation tests."""
import unittest
import numpy as np
from learning import (Environment, Grid, Observation, bfs_action, demo_grid, evaluate,
                      q_target, rollout, shortest_path, train)


class EnvironmentTests(unittest.TestCase):
    def test_blocked_move_stays_put_and_costs_five(self):
        grid = Grid(3, 2, frozenset({(1, 0)}), (0, 0), (2, 1))
        env = Environment(grid, slip=0, horizon=3)
        initial = env.reset(1)
        result = env.step(1)
        self.assertEqual(result.observation.position, initial.position)
        self.assertTrue(result.collision)
        self.assertEqual(result.reward, -5)
        self.assertEqual(result.observation.remaining, 2)
        self.assertFalse(result.done)

    def test_goal_on_last_step_is_termination_not_timeout(self):
        env = Environment(Grid(2, 1, frozenset(), (0, 0), (1, 0)), slip=0, horizon=1)
        env.reset(3)
        result = env.step(1)
        self.assertEqual(result.reward, 20)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_timeout_and_reset_restore_state_and_rng(self):
        env = Environment(demo_grid(), slip=0.3, horizon=3)
        first_obs = env.reset(81)
        first = [env.step(1) for _ in range(3)]
        self.assertTrue(first[-1].truncated)
        self.assertEqual(env.reset(81), first_obs)
        self.assertEqual([env.step(1) for _ in range(3)], first)

    def test_start_at_goal_needs_zero_actions(self):
        grid = Grid(1, 1, frozenset(), (0, 0), (0, 0))
        trial = rollout(grid, "random", 5)
        self.assertTrue(trial.success)
        self.assertEqual(trial.actions, [])
        self.assertEqual(trial.rewards, [])

    def test_invalid_probability_action_and_unreset_step_fail(self):
        with self.assertRaises(ValueError):
            Environment(demo_grid(), slip=1.1)
        env = Environment(demo_grid())
        with self.assertRaises(RuntimeError):
            env.step(1)
        env.reset(1)
        with self.assertRaises(ValueError):
            env.step(4)


class PlanningTests(unittest.TestCase):
    def test_empty_map_bfs_matches_independent_manhattan_bound(self):
        grid = Grid(5, 4, frozenset(), (0, 0), (4, 3))
        path = shortest_path(grid)
        self.assertEqual(len(path)-1, 7)
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]), 1)

    def test_disconnected_map_has_no_bfs_path(self):
        grid = Grid(3, 2, frozenset({(1, 0), (1, 1)}), (0, 0), (2, 1))
        self.assertIsNone(shortest_path(grid))
        with self.assertRaises(ValueError):
            bfs_action(grid, grid.start)

    def test_bfs_no_slip_uses_shortest_number_of_moves(self):
        grid = demo_grid()
        result = rollout(grid, "bfs", seed=7, slip=0)
        self.assertTrue(result.success)
        self.assertEqual(len(result.actions), len(shortest_path(grid))-1)
        self.assertEqual(sum(result.collisions), 0)
        self.assertTrue(all(grid.free(p) for p in result.positions))


class LearningTests(unittest.TestCase):
    def test_terminal_and_timeout_targets_do_not_bootstrap(self):
        self.assertEqual(q_target(20, 1000, 0.97, terminated=True), 20)
        self.assertEqual(q_target(-1, 1000, 0.97, truncated=True), -1)
        self.assertAlmostEqual(q_target(-1, 10, 0.9), 8)

    def test_small_problem_actually_learns_and_is_seeded(self):
        grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
        q, history = train(grid, seed=3, episodes=600, horizon=6, slip=0, log_every=100)
        repeat, same_history = train(grid, seed=3, episodes=600, horizon=6, slip=0, log_every=100)
        np.testing.assert_array_equal(q, repeat)
        self.assertEqual(history, same_history)
        self.assertGreater(np.count_nonzero(q), 0)
        self.assertEqual(q.shape, (7, 1, 3, 4))
        trial = rollout(grid, "q_learning", seed=91, q_table=q, slip=0, horizon=6)
        self.assertTrue(trial.success)
        self.assertEqual(len(trial.actions), 2)
        # The remaining=0 boundary is never trained into a nonzero value.
        np.testing.assert_array_equal(q[0], 0)

    def test_frozen_greedy_evaluation_is_reproducible_without_updates(self):
        grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
        q, _ = train(grid, seed=4, episodes=400, horizon=6, slip=0)
        before = q.copy()
        result = evaluate(grid, "q_learning", range(100001, 100021), q, slip=0.2, horizon=6)
        repeated = evaluate(grid, "q_learning", range(100001, 100021), q, slip=0.2, horizon=6)
        self.assertEqual(result, repeated)
        np.testing.assert_array_equal(q, before)
        self.assertEqual(result["evaluation_updates"], 0)
        self.assertTrue(result["q_unchanged"])
        self.assertTrue(0 <= result["success_rate"] <= 1)
        self.assertEqual(result["successes"]+result["timeouts"], 20)

    def test_invalid_evaluation_seeds_and_q_shape_fail(self):
        with self.assertRaises(ValueError):
            evaluate(demo_grid(), "random", [1, 1])
        with self.assertRaises(ValueError):
            rollout(demo_grid(), "q_learning", 1, np.zeros((1, 1, 1, 4)))


if __name__ == "__main__":
    unittest.main()
