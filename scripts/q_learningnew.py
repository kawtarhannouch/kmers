import numpy as np
import pandas as pd
from scipy.stats import poisson, gamma, nbinom
import random

def load_counts_from_file(file_path):
    with open(file_path, 'r') as f:
        counts = [int(line.strip().split()[1]) for line in f]
    return np.array(counts)

def sort_counts(counts):
    return np.sort(counts)

def compute_empirical_distribution(data):
    x_vals = np.unique(data)
    freqs = np.array([np.sum(data == x) for x in x_vals])
    probs = freqs / freqs.sum()
    return x_vals, probs

def fit_poisson(data):
    return np.mean(data)

def fit_gamma(data):
    shape, loc, scale = gamma.fit(data, floc=0)
    return shape, scale

def fit_negative_binomial(data):
    mean = np.mean(data)
    var = np.var(data)
    if var > mean:
        p = mean / var
        r = mean * p / (1 - p)
        return r, p
    else:
        return None, None
    
def compute_theoretical_mixture_pmf(x_vals, x1, x2, data):

    data_poisson = data[data <= x1]
    data_gamma = data[(data > x1) & (data <= x2)]
    data_nb = data[data > x2]

    total_len = len(data)
    w_p = len(data_poisson) / total_len
    w_g = len(data_gamma) / total_len
    w_n = len(data_nb) / total_len

    pmf_p = np.zeros_like(x_vals, dtype=float)
    pmf_g = np.zeros_like(x_vals, dtype=float)
    pmf_n = np.zeros_like(x_vals, dtype=float)

    if len(data_poisson) > 0:
        mu = fit_poisson(data_poisson)
        pmf_p = poisson.pmf(x_vals, mu)
    if len(data_gamma) > 0:
        shape, scale = fit_gamma(data_gamma)
        pmf_g = gamma.pdf(x_vals, a=shape, scale=scale)
    if len(data_nb) > 0:
        r, p = fit_negative_binomial(data_nb)
        if r is not None and p is not None:
            pmf_n = nbinom.pmf(x_vals, n=r, p=p)

    pmf_combined = w_p * pmf_p + w_g * pmf_g + w_n * pmf_n
    pmf_combined /= pmf_combined.sum()

    return pmf_combined

def compute_cross_entropy(p_empirical, q_theoretical):
    epsilon = 1e-10
    return -np.sum(p_empirical * np.log(q_theoretical + epsilon))

class QLearningAgent:
    def __init__(self, data, x_vals, p_empirical, x1_range, x2_range, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.data = data
        self.x_vals = x_vals
        self.p_empirical = p_empirical
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table as a dictionary

    def get_state(self, x1, x2):
        return (x1, x2)

    def get_possible_actions(self, x1, x2):
        actions = []
        if x1 > self.x1_range[0]:
            actions.append(('decrease_x1', x1 - 1, x2))
        if x1 < self.x1_range[1]:
            actions.append(('increase_x1', x1 + 1, x2))
        if x2 > self.x2_range[0]:
            actions.append(('decrease_x2', x1, x2 - 1))
        if x2 < self.x2_range[1]:
            actions.append(('increase_x2', x1, x2 + 1))
        return actions

    def choose_action(self, state):
        x1, x2 = state
        possible_actions = self.get_possible_actions(x1, x2)
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            return random.choice(possible_actions)
        else:
            # Exploitation
            q_values = [self.q_table.get((action[1], action[2]), 0) for action in possible_actions]
            max_q = max(q_values)
            max_actions = [action for action, q in zip(possible_actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, 0)
        max_future_q = max([self.q_table.get((a[1], a[2]), 0) for a in self.get_possible_actions(*next_state)], default=0)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state] = new_q

    def train(self, episodes):
        for episode in range(episodes):
            x1 = random.randint(self.x1_range[0], self.x1_range[1])
            x2 = random.randint(self.x2_range[0], self.x2_range[1])
            if x1 >= x2:
                continue 
            state = self.get_state(x1, x2)

            done = False
            while not done:
                action = self.choose_action(state)
                next_state = (action[1], action[2])
                q_theoretical = compute_theoretical_mixture_pmf(self.x_vals, next_state[0], next_state[1], self.data)
                cross_entropy = compute_cross_entropy(self.p_empirical, q_theoretical)
                reward = -cross_entropy  
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                done = True  

    def get_best_thresholds(self):
        best_state = max(self.q_table, key=self.q_table.get)
        return best_state
