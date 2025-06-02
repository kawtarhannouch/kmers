import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def load_and_expand_kmer_histogram(file_path: str, min_count, max_count):
    data = np.loadtxt(file_path, dtype=int)
    counts = data[:, 0]
    frequencies = data[:, 1]
    mask = (counts >= min_count) & (counts <= max_count)
    filtered_counts = counts[mask]
    filtered_frequencies = frequencies[mask]
    expanded_data = np.repeat(filtered_counts, filtered_frequencies)
    return expanded_data

def plot_poisson_qq(data, output_dir, filename='10_histo_poisson_qq_plot.png'):
    mu = data.mean()
    stats.probplot(data, dist='poisson', sparams=(mu,), plot=plt)
    plt.title('Q-Q Plot: K-mer Counts vs. Poisson Distribution')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Observed Quantiles')
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"  
    output_dir = "/data/Kaoutar/dev/py/kmers/figures_two"
    expanded_counts = load_and_expand_kmer_histogram(input_path, min_count=1, max_count=10)
    plot_poisson_qq(expanded_counts, output_dir)
