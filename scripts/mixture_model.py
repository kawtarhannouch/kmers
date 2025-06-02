import numpy as np
from scipy.stats import poisson, gamma, nbinom

class KmerMixtureModel:
    def __init__(self, counts):
        self.counts = counts
        self.pi = np.array([1/3, 1/3, 1/3])
        self.lambda_poisson = None
        self.alpha_gamma = None
        self.beta_gamma = None
        self.r_nbinom = None
        self.p_nbinom = None
        self.responsibilities = None
        self.initialize_parameters()

    def initialize_parameters(self):
        poisson_mask = (self.counts >= 1) & (self.counts <= 3)
        self.lambda_poisson = np.mean(self.counts[poisson_mask]) if np.any(poisson_mask) else 1

        gamma_mask = (self.counts >= 4) & (self.counts <= 9999)
        if np.any(gamma_mask):
            mean_gamma = np.mean(self.counts[gamma_mask])
            var_gamma = np.var(self.counts[gamma_mask])
            self.alpha_gamma = (mean_gamma ** 2) / var_gamma if var_gamma > 0 else 1
            self.beta_gamma = mean_gamma / var_gamma if var_gamma > 0 else 1
        else:
            self.alpha_gamma, self.beta_gamma = 2.0, 1.0

        nb_mask = self.counts >= 10000
        if np.any(nb_mask):
            mean_nb = np.mean(self.counts[nb_mask])
            var_nb = np.var(self.counts[nb_mask])
            if var_nb > mean_nb:
                self.r_nbinom = (mean_nb ** 2) / (var_nb - mean_nb)
                self.p_nbinom = mean_nb / var_nb
            else:
                self.r_nbinom, self.p_nbinom = 1, 0.5
        else:
            self.r_nbinom, self.p_nbinom = 1, 0.5

    def em_algorithm(self, max_iter=1000, tol=1e-6):
        log_likelihood = -np.inf
        for iteration in range(max_iter):
            poisson_probs = poisson.pmf(self.counts, self.lambda_poisson)
            gamma_probs = gamma.pdf(self.counts, self.alpha_gamma, scale=1/self.beta_gamma)
            nbinom_probs = nbinom.pmf(self.counts, self.r_nbinom, self.p_nbinom)

            total_probs = (self.pi[0] * poisson_probs +
                           self.pi[1] * gamma_probs +
                           self.pi[2] * nbinom_probs + 1e-10)

            w1 = (self.pi[0] * poisson_probs) / total_probs
            w2 = (self.pi[1] * gamma_probs) / total_probs
            w3 = (self.pi[2] * nbinom_probs) / total_probs
            self.responsibilities = np.vstack([w1, w2, w3]).T

            self.pi = self.responsibilities.mean(axis=0)
            self.lambda_poisson = np.sum(w1 * self.counts) / np.sum(w1)
            self.alpha_gamma = np.sum(w2 * self.counts) / np.sum(w2)
            self.beta_gamma = np.sum(w2) / np.sum(w2 * self.counts)
            self.r_nbinom = np.sum(w3 * self.counts) / np.sum(w3)
            self.p_nbinom = np.sum(w3) / (np.sum(w3) + np.sum(w3 * self.counts))

            new_log_likelihood = np.sum(np.log(total_probs))
            if abs(new_log_likelihood - log_likelihood) < tol:
                print(f"Converged at iteration {iteration}")
                break
            log_likelihood = new_log_likelihood

        error_posteriors = self.responsibilities[:, 0]
        artifact_posteriors = self.responsibilities[:, 2]

        valid_mask = (error_posteriors < 0.95) & (artifact_posteriors < 0.95)
        cutoff_candidates = self.counts[valid_mask]
        cutoff = np.min(cutoff_candidates) if cutoff_candidates.size > 0 else None
        print(cutoff)


def main():
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    counts = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                counts.append(int(parts[1]))

    counts = np.array(counts)
    model = KmerMixtureModel(counts)
    model.em_algorithm()

if __name__ == "__main__":
    main()
