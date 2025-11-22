"""
Brainstorming how States and Actions would work
"""

from enum import Enum
from typing import Self
import numpy as np
from maps import maps

def play_text(game_map, print_moves=False):
    """
    Play a text based version of the game
    """
    INIT_GAME(game_map)

    state = State(NEXT_GRID(drop=True), START_LOCATION)
    moves = []
    total_reward = 0
    total_moves = 0
    last_reward = 0
    while state.grid is not None:
        print("\n\n")
        # print map
        newest_n_grids = VIEW_GRIDS(current_grid=state.grid, n_grids=2)[::-1]
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



START_LOCATION = (1,1)
GAME_MAP: list[np.ndarray] | None = None
def INIT_GAME(game_map: list[np.ndarray]) -> None:
    """Initialize the game board as a list of grids"""
    global GAME_MAP
    GAME_MAP = game_map.copy()

def NEXT_GRID(drop:bool=True) -> np.ndarray | None:
    """Get the next grid in the gameboard, and (optionally) remove it from the remaining map"""
    assert GAME_MAP is not None, "Game map has not been initialized. Call INIT_GAME first."
    if len(GAME_MAP) == 0:
        return None
    if drop:
        return GAME_MAP.pop(0)
    return GAME_MAP[0]

def VIEW_GRIDS(current_grid:np.ndarray | None = None, n_grids:int = 2):
    grids = []
    if current_grid is None:
        # get n new grids
        grids = [NEXT_GRID(drop=False) for _ in range(n_grids)]
    else:
        # get current grid and n-1 bew grids
        grids = [current_grid] + [NEXT_GRID(drop=False) for _ in range(n_grids-1)]
    # drop any empty grids
    while(grids[-1] is None):
        grids = grids[:-1]
    return grids

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

class Action(Enum):
    """actions defined as (delta row, delta col) of grid"""
    LEFT = (0,-1)
    RIGHT = (0,1)
    JUMP = (-1,0)
    DUCK = (1,0)
    STAY = (0,0)


class State:
    def __init__(self, grid:np.ndarray, player_location:tuple[int,int]):
        self.grid:np.ndarray = grid
        self.player_location:tuple[int,int] = player_location

    def move(self, action:Action) -> tuple[int, Self]:
        """
        Take a given action and return the resulting reward and new state
        The new location is what is set up to take the next action.
        Ex: if current location is [1,0] and action is JUMP, then returned location is [1,0] not [0,0].
        Parameters:
            action: the action the player takes
        Returns:
            (reward, new_state): the reward and the next state the player is in
        """
        reward = self.get_reward(action)
        new_location = self._update_location(self.player_location, action)
        # after move has been done, player should no longer be jumping/ducking
        stand_row = 1
        new_location = (stand_row, new_location[1])
        if reward == 0:
            # Collision --> no next state (game over)
            print("\n\nCrash!")
            return reward, State(None, new_location)
        new_grid = NEXT_GRID(drop=True)
        return reward, State(new_grid, new_location)

    def get_reward(self, action: Action) -> int:
        """
        Given an action in the current state, return the resulting reward (0, 1).
        This is the simple version of the reward function where it does not take into account future viability.
        """
        new_location = self._update_location(self.player_location, action)
        # action results in immediate death --> no reward
        if self._collides(new_location, self.grid):
            return 0
        # no death --> reward = 1
        return 1

    def _update_location(self, player_location:tuple[int,int], action:Action):
        """
        Given a current player location and an action, return the new player location.
        Returns the location while the player is interacting with a grid (not after)
        """
        row, col = player_location
        row_change, col_change = action.value
        # update the row and column, making sure they stay in bounds of 0 and 2
        new_row = max(0, row + row_change)
        new_row = min(new_row, 2)
        new_col = max(0, col + col_change)
        new_col = min(new_col, 2)
        return (new_row, new_col)


    def _collides(self, player_location:tuple[int,int], grid:np.ndarray) -> bool:
        row, col = player_location
        if grid[row, col] == 1:
            return True
        # If the player is standing (not ducking or jumping), their height is 2 -> check both points
        # A player is standing if their location row is 1. (jumping if 0, ducking if 2)
        mid_row = 1
        bot_row = 2
        if row == mid_row and grid[bot_row, col] == 1:
            return True
        return False
    
    def __str__(self):
        return "="*50 + f"Location: {self.player_location}\nGrid:\n{self.grid}\n" + "="*50
    
if __name__ == "__main__":
    play_text(maps[1], print_moves=True)