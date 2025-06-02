import numpy as np
import random
from scipy.stats import gamma
from tqdm import trange


def load_counts_from_file(file_path):
    with open(file_path, 'r') as f:
        counts = [int(line.strip().split()[1]) for line in f]
    return np.array(counts)

def sort_counts(counts):
    return np.sort(counts)

def filter_counts_in_range(counts, x1, x2):
    return counts[(counts >= x1) & (counts <= x2)]

def compute_empirical_distribution(data):
    x_vals, freqs = np.unique(data, return_counts=True)
    probs = freqs / freqs.sum()
    return x_vals, probs

def fit_gamma_distribution(data):
    shape, loc, scale = gamma.fit(data, floc=0)
    return shape, scale

def compute_theoretical_gamma_pmf(x_vals, shape, scale):
    pdf_vals = gamma.pdf(x_vals, a=shape, loc=0, scale=scale)
    return pdf_vals / pdf_vals.sum()

def compute_cross_entropy(p_empirical, q_theoretical):
    return -np.sum(p_empirical * np.log(q_theoretical + 1e-10))

def is_valid_state(x1, x2, min_count, max_count):
    return (min_count <= x1 < max_count - 1) and (x1 < x2 <= max_count)

class KmerQAgent:
    def __init__(self, counts, min_count, max_count, alpha=0.3, gamma=0.99, epsilon=0.3):
        self.counts = counts
        self.min_count = min_count
        self.max_count = max_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
        self.q_table = self._init_q_table()

    def _init_q_table(self):
        q_table = {}
        for x1 in range(self.min_count, self.max_count):
            for x2 in range(self.min_count, self.max_count + 1):
                if is_valid_state(x1, x2, self.min_count, self.max_count):
                    q_table[(x1, x2)] = [0 for _ in self.actions]
        return q_table

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state])]

    def get_next_state(self, x1, x2, action):
        dx1, dx2 = action
        proposed_x1 = x1 + dx1
        proposed_x2 = x2 + dx2

        if is_valid_state(proposed_x1, proposed_x2, self.min_count, self.max_count):
            return (proposed_x1, proposed_x2)
        else:
            return (x1, x2)

    def get_reward(self, x1, x2):
        data = filter_counts_in_range(self.counts, x1, x2)
        x_vals, p_emp = compute_empirical_distribution(data)
        shape, scale = fit_gamma_distribution(data)
        p_gamma = compute_theoretical_gamma_pmf(x_vals, shape, scale)
        return -compute_cross_entropy(p_emp, p_gamma)

    def update_q_table(self, state, action, reward, next_state):
        a_idx = self.actions.index(action)
        max_q = max(self.q_table[next_state])
        self.q_table[state][a_idx] += self.alpha * (reward + self.gamma * max_q - self.q_table[state][a_idx])

    def train(self, episodes, steps_per_episode):
        for episode in trange(episodes, desc="Training Progress"):
            while True:
                x1 = random.randint(self.min_count, self.max_count)
                x2 = random.randint(self.min_count, self.max_count)
                if is_valid_state(x1, x2, self.min_count, self.max_count):
                    break
            state = (x1, x2)

            for step in range(steps_per_episode):
                action = self.choose_action(state)
                next_state = self.get_next_state(state[0], state[1], action)
                reward = self.get_reward(next_state[0], next_state[1])
                if step % 50 == 0:
                    print(f"Episode {episode + 1}, Step {step + 1}, State {state} → {next_state}, Reward = {reward:.6f}")
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def get_best_cutoffs(self):
        def best_q_value(state):
            return max(self.q_table[state])
        best_state = max(self.q_table, key=best_q_value)
        print("\nFinal Q-table (sample):")
        print(f"\nBest cutoff found: x1 = {best_state[0]}, x2 = {best_state[1]}")
        return best_state

if __name__ == "__main__":
    file_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    counts = load_counts_from_file(file_path)
    counts = sort_counts(counts)

    agent = KmerQAgent(counts, min_count=10, max_count=10000)
    agent.train(episodes=10000, steps_per_episode=200)
    best_cutoffs = agent.get_best_cutoffs()

            
            

