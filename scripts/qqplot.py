import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from collections import Counter

def load_and_expand_from_kmer_file(file_path: str, min_count, max_count):
    data = np.loadtxt(file_path, dtype=int, usecols=1)
    filtered_counts = data[(data >= min_count) & (data <= max_count)]
    count_frequencies = Counter(filtered_counts)
    counts = np.array(list(count_frequencies.keys()))
    print(counts[:10])
    frequencies = np.array(list(count_frequencies.values()))
    print(frequencies[:10])
    expanded_data = np.repeat(counts, frequencies)
    print("First 10 expanded counts:")
    print(expanded_data[:10])
    return expanded_data

def plot_poisson_qq(data, output_dir, filename='poisson_qq_plot.png'):
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
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"  
    output_dir = "/data/Kaoutar/dev/py/kmers/figures_two"
    expanded_counts = load_and_expand_from_kmer_file(input_path, min_count=1, max_count=10)
    plot_poisson_qq(expanded_counts, output_dir)

