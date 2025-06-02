import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random

def read_kmer_histogram(file_path):
    data = np.loadtxt(file_path, dtype=int)
    mask = data[:, 0] > 10  
    return data[mask, 0], data[mask, 1]

def expand_data(counts, frequencies, max_samples=100000, random_state=42):
    expanded = np.repeat(counts, frequencies)
    if len(expanded) > max_samples:
        random.seed(random_state)
        expanded = random.sample(list(expanded), max_samples)
    return np.array(expanded)

def fit_negative_binomial(data):
    mean_count = np.mean(data)
    var_count = np.var(data)
    print(mean_count, var_count)
    if var_count <= mean_count:
        raise ValueError("Variance must be greater than the mean for a negative binomial distribution.")
    return mean_count**2 / (var_count - mean_count), mean_count / var_count

def generate_qqplot(data, r_nb, p_nb, output_path, random_state=42):
    if len(data) > 100000:  
        random.seed(random_state)
        data = random.sample(list(data), 100000)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    stats.probplot(data, dist="nbinom", sparams=(r_nb, p_nb), plot=ax)
    ax.set_title("Q-Q Plot: Data vs Negative Binomial Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Theoretical Quantiles", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sample Quantiles", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(input_path, output_dir):
    counts, frequencies = read_kmer_histogram(input_path)
    expanded_data = expand_data(counts, frequencies)
    r_nb, p_nb = fit_negative_binomial(expanded_data)
    qqplot_nb_path = output_dir + "/qqplot_negative_binomial.png"
    generate_qqplot(expanded_data, r_nb, p_nb, qqplot_nb_path)

if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
    output_dir = "/data/Kaoutar/dev/py/kmers/figures"
    main(input_path, output_dir)

