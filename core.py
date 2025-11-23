from enum import Enum
import numpy as np

START_LOCATION = (1,1)
GAME_MAP: list[np.ndarray] | None = None
def INIT_GAME(game_map: list[np.ndarray]) -> None:
    """Initialize the game board as a list of grids"""
    global GAME_MAP
    GAME_MAP = game_map.copy()

def GRID_AT(t: int) -> np.ndarray | None:
    """Get the grid at time index t or None if past the end of the game map."""
    assert GAME_MAP is not None, "Game map has not been initialized. Call INIT_GAME first."
    if 0 <= t < len(GAME_MAP):
        return GAME_MAP[t]
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
    """A state contains a grid and the player's current location"""
    def __init__(self, time_index:int, player_location:tuple[int,int]):
        self.time_index: int = time_index
        self.player_location: tuple[int,int] = player_location
        self.grid: np.ndarray | None = GRID_AT(time_index)

    def is_terminal(self) -> bool:
        """Returns True if no more grids or the grid is None -> terminal state."""
        return self.grid is None
    
    def preview(self, action:Action) -> tuple[int, "State"]:
        """
        See the outcome (resulting reward, new state) of taking a given action.
        The new state.player_location is what is set up to take the next action.
        Ex: if current location is (1,0) and action is JUMP, then returned state's location is (1,0) not (0,0).
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
        The new location is what is set up to take the next action.
        Ex: if current location is (1,0) and action is JUMP, then returned state's location is (1,0) not (0,0).
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

        new_location = self._update_location(self.player_location, action)
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
        new_location = self._update_location(self.player_location, action)
        # after move has been done, player should no longer be jumping/ducking
        stand_row = 1
        new_location = (stand_row, new_location[1])
        if reward == 0:
            # Collision --> no next state (game over)
            print("\n\nCrash!")
            return reward, State(-1, new_location)
        return reward, State(self.time_index+1, new_location)

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
        return "="*50 + f"\nLocation: {self.player_location}\nGrid (t={self.time_index}):\n{self.grid}\n" + "="*50
