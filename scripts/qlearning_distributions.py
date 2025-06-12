import numpy as np
import random
from tqdm import trange
from scipy.stats import poisson, gamma, nbinom

def load_counts_from_file(file_path):
    with open(file_path, 'r') as f:
        counts = [int(line.strip().split()[1]) for line in f if line.strip()]
    return np.array(counts)

def sort_counts(counts):
    return np.sort(counts)

def compute_empirical_distribution(data):
    x_vals, freqs = np.unique(data, return_counts=True)
    probs = freqs / freqs.sum()
    return x_vals, probs

def compute_theoretical_pmf(x_vals, x1, x2, data):
    data_poisson = data[data <= x1]
    data_gamma   = data[(data > x1) & (data <= x2)]
    data_nbinom  = data[data > x2]
    total = len(data)

    w_p = len(data_poisson)/total 
    w_g = len(data_gamma)/total 
    w_n = len(data_nbinom)/total 

    pmf_p = poisson.pmf(x_vals, mu=np.mean(data_poisson)) if len(data_poisson)>0 else np.zeros_like(x_vals, float)
    try:
        shape, loc, scale = gamma.fit(data_gamma, floc=0)
        pmf_g = gamma.pdf(x_vals, a=shape, loc=loc, scale=scale)
    except:
        pmf_g = np.zeros_like(x_vals, float)

    mean_n = np.mean(data_nbinom) 
    var_n  = np.var(data_nbinom) 

    if len(data_nbinom)>0 and var_n>mean_n:
        r = mean_n**2 / (var_n - mean_n)
        p = mean_n / var_n
        pmf_n = nbinom.pmf(x_vals, n=r, p=p)
    else:
        pmf_n = np.zeros_like(x_vals, float)

    mixture = w_p*pmf_p + w_g*pmf_g + w_n*pmf_n
    s = mixture.sum()
    return mixture/s if s>0 else mixture

def compute_cross_entropy(p_emp, q_theor):
    eps = 1e-10
    return -np.sum(p_emp * np.log(q_theor + eps))

class KmerQAgent:
    def __init__(self, counts, min_count, max_count,
                 alpha=0.3, gamma=0.3, epsilon=0.9, increase=50, k=0.01):
        self.counts    = counts
        self.min_count = min_count
        self.max_count = max_count
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.increase  = increase
        self.k         = k

        self.actions = [
            (0, 0),
            lambda x1, x2: ((x2 - x1)//2, 0),
            lambda x1, x2: (-(x1 - self.min_count)//2, 0),
            lambda x1, x2: (0, (self.max_count - x2)//2),
            lambda x1, x2: (0, -((x2 - x1)//2)),
            lambda x1, x2: (self.increase, 0),
            lambda x1, x2: (-self.increase, 0),
            lambda x1, x2: (0, self.increase),
            lambda x1, x2: (0, -self.increase),
        ]

        self.q_table = {}
        filtered_counts = counts[(counts >= self.min_count) & (counts <= self.max_count)]
        self.x_vals, self.p_emp = compute_empirical_distribution(filtered_counts)

    def choose_action(self, state):
        self.q_table.setdefault(state, [0] * len(self.actions))
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        return int(np.argmax(self.q_table[state]))

    def get_next_state(self, x1, x2, action_id):
        action = self.actions[action_id]
        dx1, dx2 = action(x1, x2) if callable(action) else action
        new_x1 = x1 + dx1
        new_x2 = x2 + dx2
        if not (self.min_count <= new_x1 < new_x2 <= self.max_count):
            return x1, x2
        return new_x1, new_x2

    def get_reward(self, x1, x2, prev=None):
        filtered_counts = self.counts[(self.counts >= self.min_count) & (self.counts <= self.max_count)]
        q = compute_theoretical_pmf(self.x_vals, x1, x2, filtered_counts)
        reward = -compute_cross_entropy(self.p_emp, q)
        if prev:
            width_bonus = self.gamma * self.k * (x2 - x1) - self.k * (prev[1] - prev[0])
            reward += width_bonus
        return reward

    def update_q_table(self, state, action_id, reward, next_state):
        self.q_table.setdefault(state, [0] * len(self.actions))
        self.q_table.setdefault(next_state, [0] * len(self.actions))
        old_q = self.q_table[state][action_id]
        max_nq = max(self.q_table[next_state])
        self.q_table[state][action_id] += self.alpha * (reward + self.gamma * max_nq - old_q)

    def train(self, episodes, steps):
        for ep in trange(episodes, desc="Training"):
            x1 = random.randint(self.min_count, self.max_count - 2)
            x2 = random.randint(x1 + 1, self.max_count)
            state = (x1, x2)

            for st in range(steps):
                a = self.choose_action(state)
                new_state = self.get_next_state(*state, a)
                reward = self.get_reward(*new_state, prev=state)
                print(f"Ep{ep:03} Step{st:03} | x1={state[0]} x2={state[1]} → x1'={new_state[0]} x2'={new_state[1]} | Reward={reward:.6f}")
                self.update_q_table(state, a, reward, new_state)

                if new_state == state:
                    break
                state = new_state

    def best_cutoffs(self):
        best = max(self.q_table, key=lambda s: max(self.q_table[s]))
        print(f"Best cutoffs found: x1={best[0]}, x2={best[1]}")
        return best

def main():
    file_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    counts = load_counts_from_file(file_path)
    counts = sort_counts(counts)
    min_c, max_c = 1, 10000
    print(f"Empirical distribution computed over counts, range [{min_c}, {max_c}]")

    filtered_counts = counts[(counts >= min_c) & (counts <= max_c)]

    x_vals, p_emp = compute_empirical_distribution(filtered_counts)
    q_theor = compute_theoretical_pmf(x_vals, 7, 1000, filtered_counts)
    init_ce = compute_cross_entropy(p_emp, q_theor)

    print(f"Initial CE for x1=7, x2=1000: {init_ce:.4f}\n")

    agent = KmerQAgent(counts, min_c, max_c, alpha=0.3, gamma=0.3, epsilon=0.9, k=0.02)
    agent.train(episodes=2000, steps=500)
    agent.best_cutoffs()

if __name__ == "__main__":
    main()

