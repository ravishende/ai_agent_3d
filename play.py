"""
Brainstorming how States and Actions would work
"""

from enum import Enum
from typing import Self
import numpy as np
from maps import map1, map2

def main():
    game_map = map2
    INIT_GAME(game_map)

    grids = [game_map[0], game_map[1]]
    state = State(grids, START_LOCATION)

    total_reward = 0
    total_moves = 0
    while state is not None:
        print("\n\n")
        # print map
        for i, grid in enumerate(state.grids[::-1]):
            print(f"{len(grids) - i})\n{grid}\n")
        # print player
        for _ in range(2):
            print(" "*2 + " "*2*state.player_location[1] + "*")
        move = input("Choose Move (l, r, j, d, s): ")
        action = str_to_action(move)
        reward, state = state.move(action)
        total_reward += reward
        total_moves += 1

    game_over(total_reward, total_moves)



START_LOCATION = (1,1)
GAME_MAP: list[np.ndarray] | None = None
def INIT_GAME(game_map: list[np.ndarray]) -> None:
    """Initialize the game board as a list of grids"""
    global GAME_MAP
    GAME_MAP = game_map

def NEXT_GRID(drop:bool=True) -> list[np.ndarray] | None:
    """Get the next grid in the gameboard, and (optionally) remove it from the remaining map"""
    assert GAME_MAP is not None, "Game map has not been initialized. Call INIT_GAME first."
    if len(GAME_MAP) == 0:
        return None
    if drop:
        return GAME_MAP.pop(0)
    return GAME_MAP[0]

def game_over(total_reward:int, total_moves:int):
    print("\n")
    print("="*50)
    print(" "*15, "Game Over!")
    print("="*50)
    print("Total Rewards:", total_reward)
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
    # actions defined as (delta row, delta col) of grid
    LEFT = (0,-1)
    RIGHT = (0,1)
    JUMP = (-1,0)
    DUCK = (1,0)
    STAY = (0,0)


class State:
    def __init__(self, grids:list[np.ndarray], player_location:tuple[int,int]):
        self.grids:list[np.ndarray] = grids
        self.player_location:tuple[int,int] = player_location

    def move(self, action:Action) -> tuple[int, Self | None]:
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
            return reward, None
        new_grids = [self.grids[1], NEXT_GRID(drop=True)]
        # handle last state
        if new_grids is None or all(grid is None for grid in new_grids):
            return reward, None
        return reward, State(new_grids, new_location)

    def get_reward(self, action:Action) -> int:
        """
        Given an action in the current state, return the resulting reward (0, 1, or 2).
        Note: this currently only works assuming there are exactly 2 grids per state
        """
        new_location = self._update_location(self.player_location, action)
        # action results in immediate death --> no reward
        if self._collides(new_location, self.grids[0]):
            return 0
        # at least one state survivable
        total_reward = 1
        current_location = new_location
        # handle final grid of map
        if self.grids[1] is None:
            return total_reward
        
        for move in list(Action):
            new_location = self._update_location(current_location, move)
            # action results in survival --> return 2
            if not self._collides(new_location, self.grids[1]):
                return total_reward + 1
        return total_reward

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
        return "="*50 + f"Location: {self.player_location}\nGrids:\n{self.grids}\n" + "="*50
    
if __name__ == "__main__":
    main()