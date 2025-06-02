import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import mode

# Load k-mer histogram data from Jellyfish output file
histogram_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"

# Read histogram file
def load_kmer_histogram(file_path):
    kmer_histogram = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                kmer, count = int(parts[0]), int(parts[1])
                kmer_histogram[kmer] = count
    return kmer_histogram

kmer_histogram = load_kmer_histogram(histogram_file)

# Convert histogram data into k-mer count array
kmer_counts = np.array([k for k, v in kmer_histogram.items() for _ in range(v)])

# Fit a Negative Binomial distribution for true k-mers
def negative_binomial_log_likelihood(params, data):
    r, p = params
    if r <= 0 or p <= 0 or p >= 1:
        return np.inf
    return -np.sum(stats.nbinom.logpmf(data, r, p))

# Initial guesses for r and p
initial_params = [5, 0.5]
result = minimize(negative_binomial_log_likelihood, initial_params, args=(kmer_counts,), method='L-BFGS-B', bounds=[(0.1, None), (0.01, 0.99)])

# Extract fitted parameters for true k-mers
r_hat, p_hat = result.x

# Fit a Poisson model for sequencing error k-mers
lambda_poisson = np.mean([k for k in kmer_counts if k <= 2])  # Estimate based on low-frequency k-mers

# Compute probability for each k-mer count
def probability_ratio(x, lambda_poisson, r_hat, p_hat):
    p_error = stats.poisson.pmf(x, lambda_poisson)
    p_true = stats.nbinom.pmf(x, r_hat, p_hat)
    return p_true / (p_true + p_error)  # Probability of being a true k-mer

# Find cutoff where P(True k-mer) > P(Error)
k_values = np.arange(1, max(kmer_counts) + 1)
probabilities = np.array([probability_ratio(k, lambda_poisson, r_hat, p_hat) for k in k_values])
t_cutoff = k_values[np.argmax(probabilities > 0.5)]  # Smallest k where P(True) > 0.5

# Plot the histogram and fitted NB distribution with fixes
plt.figure(figsize=(10, 5))
bins = np.logspace(np.log10(1), np.log10(max(kmer_counts)), 50)  # Log-spaced bins
plt.hist(kmer_counts, bins=bins, density=True, alpha=0.6, label='Observed k-mer Counts', color='orange')
plt.axvline(t_cutoff, color='black', linestyle='dashed', label=f'Cutoff: {t_cutoff}')
plt.xlabel('K-mer Count')
plt.ylabel('Density')
plt.title('K-mer Count Distribution with Estimated Cutoff')
plt.xscale("log")  # Log-scale x-axis to handle large range
plt.yscale("log")  # Log-scale y-axis to improve visibility
plt.legend()
plt.show()

# Store results in a DataFrame
df_kmers = pd.DataFrame({'K-mer Count': kmer_counts, 'Filtered': kmer_counts > t_cutoff})

# Display cutoff result
t_cutoff