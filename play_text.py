"""
Brainstorming how States and Actions would work
"""
from core import State, Action, START_LOCATION, INIT_GAME, VIEW_GRIDS
from maps import maps

def play_text(game_map, print_moves=False):
    """
    Play a text based version of the game
    """
    INIT_GAME(game_map)

    state = State(time_index=0, player_location=START_LOCATION)
    moves = []
    total_reward = 0
    total_moves = 0
    last_reward = 0
    while state.grid is not None:
        print("\n\n")
        # print map
        newest_n_grids = VIEW_GRIDS(t=state.time_index, n_grids=2)[::-1]
        for i, grid in enumerate(newest_n_grids):
            print(f"{len(newest_n_grids) - i})\n{grid}\n")
        # print player
        for _ in range(2):
            print(" "*2 + " "*2*state.player_location[1] + "*")
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
    
if __name__ == "__main__":
    play_text(maps[1], print_moves=True)