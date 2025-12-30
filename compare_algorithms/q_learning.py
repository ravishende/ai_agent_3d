"""Q-learning that uses epsilon-greedy algorithm for action selection during training"""

import random
from typing import Dict, Tuple
from core import State, Action

ACTIONS = [
    Action.LEFT,
    Action.RIGHT,
    Action.JUMP,
    Action.DUCK,
    Action.STAY,
]

QKey = Tuple[int, int, Action]

def env_step(state: State, action: Action, trap_cols, game_map):
    if state.is_terminal():
        return state, 0.0, True

    new_row, new_col = state._update_location(state.player_col, action)
    if state._collides((new_row, new_col), state.grid_bitmap):
        return None, 0.0, True

    reward, next_state = state.move(action)
    done = next_state.is_terminal() or next_state.time_index >= len(game_map) - 1
    return next_state, reward, done


def _epsilon_greedy(q: Dict[QKey, float], state: State, epsilon: float):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: q.get((state.time_index, state.player_col, a), 0.0))


def _extract_policy(game_map, trap_cols, q: Dict[QKey, float]):
    width = game_map[0].shape[1]
    state = State(0, width // 2)
    actions = []

    for _ in range(len(game_map) * 2):
        if state.is_terminal():
            break

        action = max(ACTIONS, key=lambda a: q.get((state.time_index, state.player_col, a), 0.0))
        next_state, _, done = env_step(state, action, trap_cols, game_map)
        if next_state is None:
            break

        actions.append(action)
        state = next_state
        if done:
            break

    return actions


def q_learning(
    game_map,
    trap_cols,
    episodes: int = -1,
    alpha: float = 0.2,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    **kwargs
):
    if episodes == -1:
        episodes = len(game_map) * 1000
    q: Dict[QKey, float] = {}
    width = game_map[0].shape[1]

    for _ in range(episodes):
        state = State(0, width // 2)

        for _ in range(len(game_map) * 2):
            if state.is_terminal():
                break

            action = _epsilon_greedy(q, state, epsilon)
            next_state, reward, done = env_step(state, action, trap_cols, game_map)

            key = (state.time_index, state.player_col, action)
            old_q = q.get(key, 0.0)

            if next_state is None:
                target = reward
            else:
                target = reward + gamma * max(
                    q.get((next_state.time_index, next_state.player_col, a), 0.0)
                    for a in ACTIONS
                )

            q[key] = old_q + alpha * (target - old_q)
            if done or next_state is None:
                break

            state = next_state

    return _extract_policy(game_map, trap_cols, q)
