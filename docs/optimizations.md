# Completed and proposed optimizations

## Completed Optimizations

### Only store player_col, not player_location for states

Since each state represents where the player is in the reset stage before the next upcoming slice, the player will always be standing, therefore player_row will always be 1. So, instead of storing player_location of (1, player_col), we can just declare states with the player column.

This means, when we create our state space, we no longer need to loop over all 3 rows, we can just loop over timesteps and columns. This reduces our state space to 1/3 of the original size, meaning the agent has much less to explore.

Command: `time python3 run_rl_agent.py --length=____ --width=____ --difficulty=hard --quiet`
Sample output: `31.07s user 0.18s system 99% cpu 31.476 total`

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |  `2.496 total`   |   `1.247 total`   |    1.99 |
| 3x11x1000 | `2:21.50 total`  |  `53.337 total`   |    2.65 |
| 3x5x1000  | `1:47.60 total`  |  `31.476 total`   |    3.42 |

### Don't allow agent to use STAY action

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |  `1.247 total`   |   `1.039 total`   |    1.20 |
| 3x11x1000 |  `53.337 total`  |  `45.312 total`   |    1.69 |
| 3x5x1000  |  `31.476 total`  |  `26.371 total`   |    1.19 |

## Proposed Optimizations

### Change map slices to be bitmpas instead of numpy arrays

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |        ``        |        ``         |         |
| 3x11x1000 |        ``        |        ``         |         |
| 3x5x1000  |        ``        |        ``         |         |
