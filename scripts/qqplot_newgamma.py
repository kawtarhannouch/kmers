import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from collections import Counter

def load_and_expand_from_kmer_file(file_path: str, min_count: int, max_count: int):
    data = np.loadtxt(file_path, dtype=int, usecols=1)
    filtered_counts = data[(data >= min_count) & (data <= max_count)]
    count_frequencies = Counter(map(int,filtered_counts))
    print(count_frequencies)
    counts = np.array(list(count_frequencies.keys()))
    #print(counts[:100])
    frequencies = np.array(list(count_frequencies.values()))
    #print(frequencies[:100])
    expanded_data = np.repeat(counts, frequencies)
    return expanded_data

def plot_gamma_qq(data, output_dir, filename='gamma_qq_plot.png'):
    shape, loc, scale = stats.gamma.fit(data)
    stats.probplot(data, dist=stats.gamma, sparams=(shape, loc, scale), plot=plt)
    plt.title('Q-Q Plot: K-mer Counts vs. Gamma Distribution')
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
    expanded_counts = load_and_expand_from_kmer_file(input_path, min_count=7, max_count=10000)
    plot_gamma_qq(expanded_counts, output_dir)
