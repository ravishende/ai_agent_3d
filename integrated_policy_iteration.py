import numpy as np
from gymnasium_env import ObstacleCourseEnv
from core import GET_START_LOCATION, Action, State

class IntegratedPolicyIteration:
    def __init__(self, env: ObstacleCourseEnv, gamma=0.98):
        self.env = env
        self.gamma = gamma
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.V = np.zeros(self.n_states)
        self.policy = np.random.randint(0, self.n_actions, size=self.n_states)
        if hasattr(env, 'terminal_state_idx'):
            self.policy[env.terminal_state_idx] = 4

    def policy_evaluation(self, theta=1e-6, max_iterations=1000):
        iteration = 0
        while iteration < max_iterations:
            delta = 0

            for s in range(self.n_states):
                v_old = self.V[s]
                a = self.policy[s]
                # use expected return to handle stochastic traps (instead of preview every time)
                self.V[s] = self._expected_return(s, a)
                delta = max(delta, abs(v_old - self.V[s]))

            iteration += 1
            if delta < theta:
                break

        return iteration, delta

    def policy_improvement(self):
        policy_stable = True

        for s in range(self.n_states):
            old_action = self.policy[s]
            action_values = np.zeros(self.n_actions)

            for a in range(self.n_actions):
                action_values[a] = self._expected_return(s, a)

            best_action = np.argmax(action_values)
            self.policy[s] = best_action
            if old_action != best_action:
                policy_stable = False

        return policy_stable
    
    def _expected_return(self, state_idx: int, action_idx: int) -> float:
        """
        Compute E[ R + gamma * V(S') | s, a ] using trap_death_prob.
        """
        state = self.env._obs_to_state(state_idx)
        if state.is_terminal():
            return 0.0
        action_enum = self.env.ACTION_MAP[action_idx]
        reward_no_trap = state.get_reward(action_enum)
        # always die due to obstacle
        if reward_no_trap == 0:
            return 0.0
        _, new_col = state._update_location(state.player_col, action_enum)

        # Survival next state (trap does not trigger)
        next_state_survive = State(state.time_index + 1, new_col)
        next_idx_survive = self.env._state_to_obs(next_state_survive)

        # Probability of trap killing agent in this slice
        if new_col == state.trap_col:
            p_trap = state.trap_prob
        else:
            p_trap = 0.0

        # Expected value:
        #   with prob p_trap → terminal, reward 0, V=0
        #   with prob (1 - p_trap) → reward = 1, then continue from next_idx_survive
        expected = (1.0 - p_trap) * (reward_no_trap + self.gamma * self.V[next_idx_survive])

        return expected

    def run(self, max_iterations=100, eval_theta=1e-6, verbose=True):
        for iteration in range(max_iterations):

            eval_iters, delta = self.policy_evaluation(theta=eval_theta)
            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: Policy evaluation converged in {eval_iters} steps ({delta=})")

            policy_stable = self.policy_improvement()
            if policy_stable:
                if verbose:
                    print(f"\n✓ Policy Iteration converged in {iteration + 1} iterations!")
                return True, iteration + 1

        if verbose:
            print(f"\n⚠ Reached maximum iterations ({max_iterations})")
        return False, max_iterations

    def get_action_sequence(self, verbose=False):
        actions = []
        state_idx, _ = self.env.reset()
        if verbose:
            print("\nExtracting optimal action sequence...")
            print(f"Starting state: {GET_START_LOCATION()}")

        while True:
            action_idx = self.policy[state_idx]
            action_enum = self.env.ACTION_MAP[action_idx]
            actions.append(action_enum)

            if verbose:
                state_info = self.env.get_state_info(state_idx)
                print(f"  t={state_info['time_index']}, "
                      f"pos={state_info['position']}, "
                      f"action={action_enum.name}")

            state_idx, reward, terminated, _, _ = self.env.step(action_idx)

            if terminated:
                if verbose:
                    if reward == 0:
                        print("  → Crashed!")
                    else:
                        print("  → Successfully completed!")
                break

        return actions

    def evaluate_policy_on_map(self):
        actions = self.get_action_sequence(verbose=False)
        state_idx, _ = self.env.reset()

        total_reward = 0
        steps = 0
        success = False

        for action_enum in actions:
            action_idx = self.env.ACTION_TO_IDX[action_enum]
            state_idx, reward, terminated, _, _ = self.env.step(action_idx)
            total_reward += reward
            steps += 1

            if terminated:
                success = reward > 0
                break

        return {
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'actions': actions
        }


def solve_map(game_map, map_width=None, gamma=0.98, verbose=True) -> tuple[list[Action], IntegratedPolicyIteration]:
    # Create environment
    env = ObstacleCourseEnv(game_map, num_cols=map_width)

    # Run policy iteration
    pi = IntegratedPolicyIteration(env, gamma=gamma)
    converged, iterations = pi.run(verbose=verbose)

    # Get optimal action sequence
    optimal_actions = pi.get_action_sequence(verbose=verbose)

    if verbose:
        print(f"\nOptimal action sequence length: {len(optimal_actions)}")
        print("Actions:", [a.name for a in optimal_actions])

    return optimal_actions, pi
