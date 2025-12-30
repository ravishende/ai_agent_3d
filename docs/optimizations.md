# Completed Optimizations

## Only store player_col, not player_location for states

Since each state represents where the player is in the reset stage before the next upcoming slice, the player will always be standing, therefore player_row will always be 1. So, instead of storing player_location of (1, player_col), we can just declare states with the player column.

This means, when we create our state space, we no longer need to loop over all 3 rows, we can just loop over timesteps and columns. This reduces our state space to 1/3 of the original size, meaning the agent has much less to explore.

Command: `time python3 run_rl_agent.py --length=____ --width=____ --difficulty=hard --quiet`
Sample output: `31.07s user 0.18s system 99% cpu 31.476 total`

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |  `2.496 total`   |   `1.247 total`   |    1.99 |
| 3x11x1000 | `2:21.50 total`  |  `53.337 total`   |    2.65 |
| 3x5x1000  | `1:47.60 total`  |  `31.476 total`   |    3.42 |

## Don't allow agent to use STAY action

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |  `1.247 total`   |   `1.039 total`   |    1.20 |
| 3x11x1000 |  `53.337 total`  |  `45.312 total`   |    1.69 |
| 3x5x1000  |  `31.476 total`  |  `26.371 total`   |    1.19 |

## Change map slices to be bitmaps instead of numpy arrays (in map generation and usage)

Note: Before this optimization, there was the introduction of traps to the codebase. Therefore, the command has changed to ignore traps.
Command: `time python3 run_rl_agent.py --length=____ --width=____ --difficulty=hard --quiet --trap_spawn_prob=0`.

There was not much noticeable speedup from this optimization due to the bottleneck not being memory lookups or insertions in grids. If the memory IO of numpy 2D arrays were more of a bottleneck, then this optimization may have been more useful. We do seem to see this more in the largest map: 3x11x1000 where the speedup is 1.28.

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |  `1.074 total`   |   `1.095 total`   |    0.98 |
| 3x11x1000 | `1:34.33 total`  |  `1:13.43 total`  |    1.28 |
| 3x5x1000  |  `20.495 total`  |  `21.701 total`   |    0.94 |

## Total Speedup

Here is the speedup comparing the original (no optimzations) to all 3 optimizations being implemented.
Note that, as with all previous optimization tables, the pre-optimization times are run immediately prior to the post-optimization by checking out previous commits. This way, factors of varying computer speeds by day/application are mitigated.

| Map       | Pre-optimization | Post-optimization | Speedup |
| :-------- | :--------------: | :---------------: | ------: |
| 3x11x100  |  `2.903 total`   |   `1.095 total`   |    2.65 |
| 3x11x1000 | `3:37.94 total`  |  `1:13.43 total`  |    2.97 |
| 3x5x1000  | `1:47.60 total`  |  `1:03.68 total`  |    1.69 |
