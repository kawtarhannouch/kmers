import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize


def load_kmer_histogram(file_path):
    kmer_histogram = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                kmer, count = int(parts[0]), int(parts[1])
                kmer_histogram[kmer] = count
    return kmer_histogram

def poisson_probability(k, lambda_poisson):
    return stats.poisson.pmf(k, lambda_poisson)

def negative_binomial_probability(k, r_hat, p_hat):
    return stats.nbinom.pmf(k, r_hat, p_hat)

def negative_binomial_log_likelihood(params, data):
    r, p = params
    if r <= 0 or p < 0 or p > 1:
        return np.inf
    return -np.sum(stats.nbinom.logpmf(data, r, p))

def probability_ratio(k, lambda_poisson, r_hat, p_hat):
    p_error = poisson_probability(k, lambda_poisson)  
    p_true = negative_binomial_probability(k, r_hat, p_hat)  
    return p_true / (p_true + p_error)  

def estimate_cutoff(kmer_counts):
    kmer_counts_nb = [k for k in kmer_counts if k > 2]
    initial_params = [5, 0.5]
    result = minimize(negative_binomial_log_likelihood, initial_params, 
                      args=(kmer_counts_nb,), method='L-BFGS-B', 
                      bounds=[(0.1, None), (0.01, 0.99)])

    r_hat, p_hat = result.x  
    lambda_poisson = np.mean([k for k in kmer_counts if k <= 2]) 
    k_values = np.arange(1, max(kmer_counts) + 1)

    """for i, k in enumerate(k_values):
        print(k)
        if i == 9:  
           break  """

    probabilities = np.array([probability_ratio(k, lambda_poisson, r_hat, p_hat) for k in k_values])
    t_cutoff = k_values[np.argmax(probabilities > 0.5)]
    
    return t_cutoff, r_hat, p_hat, lambda_poisson

def plot_kmer_histogram_kde(kmer_counts, t_cutoff):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(min(kmer_counts), max(kmer_counts), 50)  
    plt.hist(kmer_counts, bins=bins, alpha=0.4, color="#D55E00", 
             edgecolor="black", linewidth=1.2, label="Histogram")

    sns.kdeplot(kmer_counts, color="blue", linewidth=2, label="KDE Density")
    plt.axvline(t_cutoff, color="black", linestyle="dashed", linewidth=2, label=f"Cutoff: {t_cutoff}")
    plt.xlabel("K-mer Count", fontsize=16, fontweight="bold", labelpad=10)
    plt.ylabel("Density / Count", fontsize=16, fontweight="bold", labelpad=10)
    plt.title("K-mer Count Histogram & KDE with Estimated Cutoff", fontsize=18, fontweight="bold", pad=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False, loc="upper right")
    for fmt in ["pdf", "png"]:
        plt.savefig(f"kmer_histogram_kde.{fmt}", format=fmt, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    histogram_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
    kmer_histogram = load_kmer_histogram(histogram_file)
    kmer_counts = np.array([k for k, v in kmer_histogram.items() for _ in range(v)])
    t_cutoff, r_hat, p_hat, lambda_poisson = estimate_cutoff(kmer_counts)
    plot_kmer_histogram_kde(kmer_counts, t_cutoff)
    df_kmers = pd.DataFrame({'K-mer Count': kmer_counts, 'Filtered': kmer_counts > t_cutoff})
    return df_kmers, t_cutoff

if __name__ == "__main__":
    df_kmers, t_cutoff = main()
