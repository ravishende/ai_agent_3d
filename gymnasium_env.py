
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core import State, Action, INIT_GAME, GET_START_LOCATION


class ObstacleCourseEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    # Map Action enum to integer indices
    ACTION_MAP = {
        0: Action.LEFT,
        1: Action.RIGHT,
        2: Action.JUMP,
        3: Action.DUCK,
        4: Action.STAY
    }

    ACTION_TO_IDX = {v: k for k, v in ACTION_MAP.items()}

    def __init__(self, game_map: list[np.ndarray]):
        super().__init__()

        self.game_map = game_map
        INIT_GAME(game_map)

        self.num_timesteps = len(game_map)
        self.num_rows = 3
        self.num_cols = game_map[0].shape[1]

        self.action_space = spaces.Discrete(5)

        self.num_states = (self.num_timesteps + 1) * self.num_cols + 1
        self.observation_space = spaces.Discrete(self.num_states)

        self.state = None

        self._init_state_mappings()

    def _init_state_mappings(self):
        """Initialize bidirectional mappings between State objects and integer indices."""
        self.state_to_idx = {}
        self.idx_to_state = {}

        idx = 0
        for t in range(self.num_timesteps + 1):
            for col in range(self.num_cols):
                state_key = (t, col)
                self.state_to_idx[state_key] = idx
                self.idx_to_state[idx] = state_key
                idx += 1

        self.terminal_state_idx = idx
        self.state_to_idx[(-1, -1)] = self.terminal_state_idx
        self.idx_to_state[self.terminal_state_idx] = (-1, -1)

    def _state_to_obs(self, state: State) -> int:
        """Convert a State object to an observation (integer index)."""
        if state.is_terminal() and state.time_index == -1:
            return self.terminal_state_idx

        state_key = (state.time_index, state.player_col)
        return self.state_to_idx[state_key]

    def _obs_to_state(self, obs: int) -> State:
        """Convert an observation (integer index) to a State object."""
        if obs == self.terminal_state_idx:
            return State(-1, -1)

        t, col = self.idx_to_state[obs]
        return State(t, col)

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        self.state = State(time_index=0, player_col=GET_START_LOCATION()[1])
        obs = self._state_to_obs(self.state)

        return obs, {}

    def step(self, action: int):

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action_enum = self.ACTION_MAP[action]

        reward, new_state = self.state.move(action_enum)

        self.state = new_state

        obs = self._state_to_obs(new_state)

        terminated = new_state.is_terminal()
        truncated = False

        info = {
            'time_index': new_state.time_index,
            'player_location': new_state.player_col,
            'action_taken': action_enum.name
        }

        return obs, reward, terminated, truncated, info

    def get_state_info(self, state_idx: int) -> dict:
        """Get human-readable information about a state."""
        if state_idx == self.terminal_state_idx:
            return {
                'time_index': -1,
                'position': 'TERMINAL',
                'grid': None
            }

        t, col = self.idx_to_state[state_idx]
        state = State(t, col)

        return {
            'time_index': t,
            'player_col': col,
            'grid': state.grid
        }

    def preview_action(self, state_idx: int, action: int) -> tuple[int, int, bool]:

        state = self._obs_to_state(state_idx)
        action_enum = self.ACTION_MAP[action]

        reward, next_state = state.preview(action_enum)
        next_state_idx = self._state_to_obs(next_state)
        done = next_state.is_terminal()

        return next_state_idx, reward, done

    def render(self):
        if self.state is None:
            print("Environment not initialized.")
            return

        print(f"\n{'='*50}")
        print(f"Time: {self.state.time_index}")
        print(f"Player Location: {self.state.player_col}")
        if self.state.grid is not None:
            print(f"Current Grid:\n{self.state.grid}")
        else:
            print("TERMINAL STATE")
        print(f"{'='*50}\n")

    def close(self):
        """Clean up resources."""
        pass
