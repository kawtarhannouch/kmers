import numpy as np
from scipy.stats import poisson, gamma, nbinom
from collections import Counter
import random
from tqdm import trange

# Load and preprocess data
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

# Fitting distributions
def fit_poisson(data):
    return np.mean(data)

def fit_gamma(data):
    shape, loc, scale = gamma.fit(data, floc=0)
    return shape, scale

def fit_negative_binomial(data):
    mean, var = np.mean(data), np.var(data)
    if var > mean:
        p = mean / var
        r = mean * p / (1 - p)
        return r, p
    return None, None

# Mixture model and cross-entropy
def compute_theoretical_mixture_pmf(x_vals, x1, x2, data):
    data_p = data[data <= x1]
    data_g = data[(data > x1) & (data <= x2)]
    data_n = data[data > x2]
    total = len(data)
    w_p, w_g, w_n = len(data_p)/total, len(data_g)/total, len(data_n)/total
    pmf_p = poisson.pmf(x_vals, fit_poisson(data_p)) if len(data_p) else np.zeros_like(x_vals)
    if len(data_g):
        shape, scale = fit_gamma(data_g)
        pmf_g = gamma.pdf(x_vals, a=shape, scale=scale)
    else:
        pmf_g = np.zeros_like(x_vals)
    if len(data_n):
        r, p = fit_negative_binomial(data_n)
        pmf_n = nbinom.pmf(x_vals, n=r, p=p) if r is not None else np.zeros_like(x_vals)
    else:
        pmf_n = np.zeros_like(x_vals)
    pmf = w_p * pmf_p + w_g * pmf_g + w_n * pmf_n
    return pmf / pmf.sum()

def compute_cross_entropy(p_empirical, q_theoretical):
    eps = 1e-10
    return -np.sum(p_empirical * np.log(q_theoretical + eps))

def is_valid_state(x1, x2, min_c, max_c):
    return min_c <= x1 < x2 <= max_c

# Standalone actions
def action_stop(x1, x2, min_c, max_c, inc): return (0, 0)
def action_dich_x1_increase(x1, x2, min_c, max_c, inc): return ((x2 - x1) / 2, 0)
def action_dich_x1_decrease(x1, x2, min_c, max_c, inc): return (-(x1 - min_c) / 2, 0)
def action_dich_x2_increase(x1, x2, min_c, max_c, inc): return (0, (max_c - x2) / 2)
def action_dich_x2_decrease(x1, x2, min_c, max_c, inc): return (0, -(x2 - x1) / 2)
def action_slight_x1_increase(x1, x2, min_c, max_c, inc): return (inc, 0)
def action_slight_x1_decrease(x1, x2, min_c, max_c, inc): return (-inc, 0)
def action_slight_x2_increase(x1, x2, min_c, max_c, inc): return (0, inc)
def action_slight_x2_decrease(x1, x2, min_c, max_c, inc): return (0, -inc)

actions = [
    action_stop,
    action_dich_x1_increase,
    action_dich_x1_decrease,
    action_dich_x2_increase,
    action_dich_x2_decrease,
    action_slight_x1_increase,
    action_slight_x1_decrease,
    action_slight_x2_increase,
    action_slight_x2_decrease
]

# Q-learning agent
class KmerQAgent:
    def __init__(self, counts, min_count, max_count, increase=10, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.counts = counts
        self.min_count = min_count
        self.max_count = max_count
        self.increase = increase
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.x_vals, self.p_emp = compute_empirical_distribution(counts)
        self.q_table = {}
        self.num_actions = len(actions)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.num_actions
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0, self.num_actions-1)
        return np.argmax(self.q_table[state])

    def get_next_state(self, x1, x2, action_idx):
        dx1, dx2 = actions[action_idx](x1, x2, self.min_count, self.max_count, self.increase)
        x1_new = max(self.min_count, min(x1 + dx1, self.max_count))
        x2_new = max(self.min_count, min(x2 + dx2, self.max_count))
        return (x1_new, x2_new) if is_valid_state(x1_new, x2_new, self.min_count, self.max_count) else (x1, x2)

    def get_reward(self, x1, x2):
        q_theory = compute_theoretical_mixture_pmf(self.x_vals, x1, x2, self.counts)
        return -compute_cross_entropy(self.p_emp, q_theory)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.num_actions
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * self.num_actions
        max_q = max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_q - self.q_table[state][action])

    def train(self, episodes, steps_per_episode):
        for _ in trange(episodes, desc="Training Progress"):
            x1 = random.randint(self.min_count, self.max_count-1)
            x2 = random.randint(self.min_count+1, self.max_count)
            if not is_valid_state(x1,x2,self.min_count,self.max_count): continue
            state = (x1,x2)
            for _ in range(steps_per_episode):
                action = self.choose_action(state)
                next_state = self.get_next_state(state[0], state[1], action)
                reward = self.get_reward(*next_state)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def get_best_cutoffs(self):
        if not self.q_table: return None
        best_state = max(self.q_table, key=lambda s: max(self.q_table[s]))
        print(f"Best cutoffs found: x1={best_state[0]}, x2={best_state[1]}")
        return best_state

if __name__=="__main__":
    file_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    counts = load_counts_from_file(file_path)
    counts = sort_counts(counts)
    agent = KmerQAgent(counts, min_count=1, max_count=20000)
    agent.train(episodes=300, steps_per_episode=200)
    best_cutoffs = agent.get_best_cutoffs()
