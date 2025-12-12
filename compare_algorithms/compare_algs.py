import time
import argparse
import pandas as pd
import sys
import os
# change path to parent directory's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from maps import maps
from map_generator import MapGenerator
from run_rl_agent import train_rl
from dfs import dfs_allow_traps, dfs_avoid_traps
from q_learning import q_learning
from core import INIT_GAME, GET_START_LOCATION, State

pd.set_option("display.max_columns", None)

def run_and_get_results(algorithm, game_map, trap_cols, start_state, n_runs, alg_name:str):
    train_start = time.time()
    actions = algorithm(game_map=game_map, trap_cols=trap_cols)
    train_end = time.time()
    # most algs will return a list of actions. some return a statistics dict --> get actions
    if isinstance(actions, dict):
        actions = actions["actions"]

    rewards = []
    n_successes = 0
    for _ in range(n_runs):
        total_reward, _ = start_state.preview_sequence(actions)
        if total_reward == len(game_map):
            n_successes += 1
        rewards.append(total_reward)

    
    run_stats = {
        "map_length": len(game_map),
        "map_width": game_map[0].shape[1],
        "n_runs": n_runs,
        "algorithm": alg_name,
        "elapsed_time": train_end - train_start,
        "avg_reward": sum(rewards) / len(rewards),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "success_rate": n_successes / n_runs,
        "n_successes": n_successes,
        "actions": actions,
    }
    return run_stats

def main(map_width, map_length, difficulty, trap_death_prob, trap_spawn_prob, n_runs=100, n_maps=5):
    map_generator = MapGenerator(n_cols=map_width)
    game_map, trap_cols = map_generator.generate_track(
        map_length, trap_spawn_prob=trap_spawn_prob, difficulty=difficulty)
    INIT_GAME(game_map, trap_cols, trap_death_prob)
    start_col = GET_START_LOCATION()[1]
    start_state = State(0, start_col)
    alg_run_stats = []
    alg_run_stats.append(run_and_get_results(
        algorithm=train_rl,
        game_map=game_map,
        trap_cols=trap_cols,
        start_state=start_state,
        n_runs=n_runs,
        alg_name="agent"
    ))
    alg_run_stats.append(run_and_get_results(
        algorithm=dfs_avoid_traps,
        game_map=game_map,
        trap_cols=trap_cols,
        start_state=start_state,
        n_runs=n_runs,
        alg_name="dfs_avoid_traps"
    ))
    alg_run_stats.append(run_and_get_results(
        algorithm=dfs_allow_traps,
        game_map=game_map,
        trap_cols=trap_cols,
        start_state=start_state,
        n_runs=n_runs,
        alg_name="dfs_allow_traps"
    ))
    alg_run_stats.append(run_and_get_results(
        algorithm=q_learning,
        game_map=game_map,
        trap_cols=trap_cols,
        start_state=start_state,
        n_runs=n_runs,
        alg_name="q_learning"
    ))
    
    df = pd.DataFrame(alg_run_stats)
    print(df)
    df.to_csv("alg_run_stats3.csv", index=False)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent to navigate obstacle course"
    )
    
    parser.add_argument(
        "--map",
        type=int,
        default=1,
        help=f"Which fixed map to use (1-{len(maps)})"
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use a fixed map instead of generating one"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard", "expert"],
        help="Difficulty of generated map"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=20,
        help="Length of generated map (timesteps)"
    )

    parser.add_argument(
        "--width", 
        type=int,
        default=3,
        help="Width of the generated track (default: 3)"
    )
    parser.add_argument(
        "--trap_death_prob", "--trap-death-prob", "--trapdeathprob", 
        type=float,
        default=0.3,
        help="Probability of a trap activating when the player steps on it -- 0 <= trap_prob < 1"
    )
    parser.add_argument(
        "--trap_spawn_prob", "--trap-spawn-prob", "--trapspawnprob",
        type=float,
        default=0.3,
        help="Probability of traps spawning in any given slice (controls how many traps there are)"
    )
    
    args = parser.parse_args()
    return args

if __name__ ==  "__main__":
    args = get_args()
    main(
        map_width=args.width,
        map_length=args.length,
        difficulty=args.difficulty,
        trap_death_prob=args.trap_death_prob,
        trap_spawn_prob=args.trap_spawn_prob
    )
