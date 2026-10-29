# Model card: finite-horizon Q-table navigation

## Intended use

Education and reproducible experiments about reinforcement learning, stochastic actions, reward tradeoffs and test isolation. This is a single-map, discrete grid simulation. It is not for physical robot control, safety decisions or claims of real-world autonomy.

## Model and data

A tabular Q learner uses `(remaining_steps, y, x)` and four actions. The saved primary table has shape `(51, 7, 9, 4)`, starts at zero and is trained by online temporal-difference updates. Its map, start/goal and metadata are included in `q_table.npz`; load numeric arrays with `allow_pickle=False`.

The map is a hand-specified 9 × 7 synthetic grid with nine obstacles, fixed start `(0, 0)` and goal `(8, 6)`. No external or personal data is used. Actions slip to perpendicular directions; blocked moves stay in place. Rewards are +20 at goal, −1 for other free moves, −5 for blocked attempts. There are at most 50 actions; goal arrival on the last action is a true termination, not a timeout. Neither terminal nor timeout transitions bootstrap.

Training uses 12,000 episodes per seed, slip 0.10, learning rate 0.25, discount 0.97, epsilon 0.90 decaying to 0.05, and seeds 7, 19 and 43. Seed 7 is the declared primary model, not the best-performing selected seed. The primary table has 8,800 nonzero entries in the recorded run. The finite budget and fixed learning rate do not establish convergence or optimality.

## Evaluation evidence

All models are evaluated without updates or exploration on the same 100 held-out episode seeds, at slip 0, 0.10, 0.30 and 0.50. The map is not held out. Evaluation uses a read-only table view and checks an unchanged snapshot. Shared noise streams support fair paired policy comparisons; random action choices use a separate RNG. BFS replans after each movement but ignores action slip and rewards, so it is not a stochastic-optimal reference.

Primary success counts are 100/100 at 0%, 10% and 30% slip, and 88/100 at 50%. Nominal mean return is −3.36, compared with BFS 1.79 and random −96.49. At 50% slip, Q gets mean return −37.88 with 12 timeouts; BFS gets −46.91 with 5 timeouts. Reported returns are undiscounted. Q's noiseless route takes 22 moves, while the deterministic shortest route has 14. These results demonstrate useful learning and clear suboptimality.

Three training seeds produce 87–88 successes out of 100 at the 50% stress condition. The 95% Wilson success intervals in `metrics.json` cover episode variation for a fixed trained policy, not general training uncertainty or new-map performance. Full per-episode results and parameters are retained.

## Limits

No new-map generalization, continuous geometry, sensor noise, localization error, moving obstacles, path clearance, robot footprint, acceleration, actuator dynamics, neural policy, ROS integration or hardware testing is implemented. Collision counts are blocked discrete attempts; the environment prevents entry into obstacles. The inspected tests are not a permanent source of untouched data for future tuning.

## Reproducibility and provenance

`python -m unittest -v` passes all 12 local tests. `python demo.py` regenerates the experiment. The PNG was visually inspected and the numeric checkpoint can be loaded for independent frozen evaluation. Recorded versions are Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0. This does not assert a successful remote CI run.

AI-assisted initial educational implementation prepared for Nishanth Senthil Kumar; executed locally. Independent authorship, mastery, professional deployment and hardware work are not implied. The owner should reproduce, understand and extend the work before describing personal engineering contributions.
