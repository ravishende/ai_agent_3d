"""
Brainstorming how States and Actions would work
"""
import random
from core import State, Action, GET_START_LOCATION, INIT_GAME, VIEW_GRIDS
from maps import maps

def play_text(game_map, trap_cols=None, trap_death_prob=0.3, print_moves=False, difficulty="medium"):
    """
    Play a text based version of the game
    """
    if trap_cols is None:
        trap_cols = [-1 for _ in range(len(game_map))]
    INIT_GAME(game_map, trap_cols, trap_death_prob)

    state = State(time_index=0, player_col=GET_START_LOCATION()[1])
    moves = []
    total_reward = 0
    total_moves = 0
    last_reward = 0
    while state.grid is not None:
        print("\n\n")
        # print map
        difficulties = ["easy", "medium", "hard", "expert"]
        n_grids = 2 + difficulties.index(difficulty)  # for harder modes, show further into the future
        newest_n_grids, newest_n_traps = VIEW_GRIDS(t=state.time_index, n_grids=n_grids)
        # reverse order of lists so they display closest first
        newest_n_grids = newest_n_grids[::-1]
        newest_n_traps = newest_n_traps[::-1]
        for i, grid in enumerate(newest_n_grids):
            # grid
            print(f"{len(newest_n_grids) - i})\n{grid}")
            # trap
            trap_col = newest_n_traps[i]
            if trap_col != -1:
                print(" "*2 + " "*2*trap_col + "x")
            print("\n")
        # print player
        for _ in range(2):
            print(" "*2 + " "*2*state.player_col + "*")
        move = input("Choose Move (l, r, j, d, s): ")
        moves.append(move)
        action = str_to_action(move)
        reward, state = state.move(action)
        total_reward += reward
        total_moves += 1
        last_reward = reward

    won = last_reward > 0
    game_over(total_reward, total_moves, won=won)
    if print_moves:
        print("Moves =", moves)
    return moves

def game_over(total_reward:int, total_moves:int, won=False):
    print("\n")
    print("="*50)
    if won:
        print(" "*15, "You Win!")
    else:
        print(" "*15, "Game Over!")
    print("="*50)
    print("Total Reward:", total_reward)
    print("Total Moves:", total_moves, "\n\n")

def str_to_action(move:str):
    "Given a move (l, r, j, d, s), return an Action"
    actions = {
        "l": Action.LEFT,
        "r": Action.RIGHT,
        "j": Action.JUMP,
        "d": Action.DUCK,
        "s": Action.STAY
    }
    move = move.lower()
    if move not in actions:
        raise KeyError(f"Chosen move not in accepted moves: {list(actions.keys())}")
    return actions[move]

def main():
    print("="*50)
    print("run through an obstacle course, avoiding obstacles (1) and trying to avoid traps (x).")
    print("You are 2 tall when standing, and 1 tall when jumping or ducking.")
    print("Grids are numbered in closeness to you. Good luck!")
    print("="*50)
    game_map = maps[1]
    trap_cols = [random.randint(0, 2) for _ in range(len(game_map))]
    play_text(game_map, trap_cols, print_moves=True)

if __name__ == "__main__":
    main()