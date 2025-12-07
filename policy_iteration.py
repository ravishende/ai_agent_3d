import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=.098):
        self.env = env
        self.gamma = gamma
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.v = {s: 0.0 for s in self.states} 
        self.policy = {s: np.random.choice(self.actions) for s in self.states} 

    def next_state(self, state, action):
        backup = self.env.state
        self.env.state = state
        next_state, reward, done = self.env.step(action)
        self.env.state = backup
        return next_state, reward, done
    
    def policy_evaluation(self, theta=1e-10):
        while True:
            delta = 0
            for s in self.states:
                v = self.v[s]
                a = self.policy[s]
                next_state, reward, done = self.next_state(s, a)
                self.v[s] = reward + self.gamma * (0 if done else self.v[next_state])
                delta = max(delta, abs(v - self.v[s]))
            if delta < theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in self.states:
            old_a = self.policy[s]
            action_values = {}
            for a in self.actions:
                next_state, reward, done = self.next_state(s, a)
                action_values[a] = reward + self.gamma * (0 if done else self.v[next_state])
            best_action = max(action_values, key=action_values.get)
            self.policy[s] = best_action
            if old_a != best_action:
                policy_stable = False
        return policy_stable