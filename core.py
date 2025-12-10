from enum import Enum
import numpy as np

_GAME_MAP: list[np.ndarray] | None = None
_START_LOCATION = (1,1) # for 3xN maps, call START_LOCATION
_MAP_WIDTH = 3
_STAND_ROW = 1

def INIT_START_LOCATION(map_width: int):
    """Call this function before using START_LOCATION on 3xN maps
    Parameters:
        map_width: how many columns are in a map
    """
    global _START_LOCATION, _MAP_WIDTH
    start_col = map_width // 2
    _START_LOCATION = (_STAND_ROW, start_col)
    _MAP_WIDTH = map_width


def GET_START_LOCATION():
    return _START_LOCATION


def INIT_GAME(game_map: list[np.ndarray]) -> None:
    """
    Initialize the game board as a list of grids. (and init START_LOCATION)
    """
    global _GAME_MAP, _MAP_WIDTH
    _GAME_MAP = game_map.copy()
    map_width = game_map[0].shape[1]
    _MAP_WIDTH = map_width
    INIT_START_LOCATION(map_width)

def GRID_AT(t: int) -> np.ndarray | None:
    """Get the grid at time index t or None if past the end of the game map."""
    assert _GAME_MAP is not None, "Game map has not been initialized. Call INIT_GAME first."
    if 0 <= t < len(_GAME_MAP):
        return _GAME_MAP[t]
    return None

def VIEW_GRIDS(t: int, n_grids:int = 2):
    """View up to n future grids starting from time index t."""
    grids = []
    for offset in range(n_grids):
        grid = GRID_AT(t+offset)
        if grid is None:
            break
        grids.append(grid)
    return grids


class Action(Enum):
    """actions defined as (delta row, delta col) of grid"""
    LEFT = (0,-1)
    RIGHT = (0,1)
    JUMP = (-1,0)
    DUCK = (1,0)
    STAY = (0,0)


class State:
    """
    A state contains a grid and the player's current column
    We don't need the row because the player is always standing before the next slice comes (row=1)
    """
    def __init__(self, time_index:int, player_col:int):
        self.time_index: int = time_index
        self.player_col: int = player_col
        self.grid: np.ndarray | None = GRID_AT(time_index)

    def is_terminal(self) -> bool:
        """Returns True if no more grids or the grid is None -> terminal state."""
        return self.grid is None
    
    def preview(self, action:Action) -> tuple[int, "State"]:
        """
        See the outcome (resulting reward, new state) of taking a given action.
        If the action is JUMP, STAY, or DUCK, the new state.player_col is the same. 
        LEFT and RIGHT actions can change the resulting state.player_col.
        Parameters:
            action: the action the player takes
        Returns:
            (reward, new_state): the reward and the next state the player is in
        Note: currently `preview()` == `move()`, but in the future, `move()` may have side effects
        """
        return self._transition(action)

    def move(self, action:Action) -> tuple[int, "State"]:
        """
        Take a given action and return the resulting reward and new state
        If the action is JUMP, STAY, or DUCK, the new state.player_col is the same. 
        LEFT and RIGHT actions can change the resulting state.player_col.
        Parameters:
            action: the action the player takes
        Returns:
            (reward, new_state): the reward and the next state the player is in
        Note: currently `preview()` == `move()`, but in the future, `move()` may have side effects
        """
        return self._transition(action)

    def get_reward(self, action: Action) -> int:
        """
        Given an action in the current state, return the resulting reward (0, 1).
        This is the simple version of the reward function where it does not take into account future viability.
        """
        assert self.grid is not None, "Cannnot perform more actions from a terminal state."

        new_location = self._update_location(self.player_col, action)
        # action results in immediate death --> no reward
        if self._collides(new_location, self.grid):
            return 0
        # no death --> reward = 1
        return 1
    
    def preview_sequence(self, actions: list[Action]) -> tuple[int, "State"]:
        """
        Preview reward and resulting state of multiple actions.
        Returns (total_reward, final_state) without changing the real state.
        """
        total_reward = 0
        state = self
        for action in actions:
            reward, state = state.move(action)
            total_reward += reward
            if state.is_terminal():
                break
        return total_reward, state
    
    def _transition(self, action:Action) -> tuple[int, "State"]:
        """Return the resulting reward and new state from taking a given action"""
        if self.is_terminal():
            # already in a terminal state, no need to change state
            return 0, self

        reward = self.get_reward(action)
        _, new_col = self._update_location(self.player_col, action)
        # after move has been done, player should no longer be jumping/ducking
        if reward == 0:
            # Collision --> no next state (game over)
            return reward, State(-1, new_col)
        return reward, State(self.time_index+1, new_col)

    def _update_location(self, player_col:int, action:Action) -> tuple[int, int]:
        """
        Given a current player location and an action, return the player location during the action.
        Returns the location while the player is interacting with a grid (not after)
        """
        col = player_col
        row = 1  # the player is always standing before the coming slice (row=1)
        row_change, col_change = action.value
        # update the row and column, making sure they stay in bounds of 0->2 and 0->_MAP_WIDTH-1
        max_row = 2
        new_row = max(0, row + row_change)
        new_row = min(new_row, max_row)
        new_col = max(0, col + col_change)
        new_col = min(new_col, _MAP_WIDTH-1)
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
        return "="*50 + f"\nLocation: col={self.player_col}\nGrid (t={self.time_index}):\n{self.grid}\n" + "="*50
