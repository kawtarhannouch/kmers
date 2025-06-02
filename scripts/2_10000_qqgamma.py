import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter

def load_and_aggregate_counts(file_path, min_threshold=2, max_threshold=10000):
    counts = np.loadtxt(file_path, dtype=int, usecols=1)  
    filtered_counts = counts[(counts > min_threshold) & (counts <= max_threshold)]
    print(f"Max count after filtering: {np.max(filtered_counts)}")
    count_freq = Counter(filtered_counts) 
    for i, (count, freq) in enumerate(count_freq.items()):
        print(f"{count}: {freq}")
        if i == 4:  
           break 

    unique_counts = np.array(list(count_freq.keys())) 
    frequencies = np.array(list(count_freq.values()))  
    #print(f"First value in unique_counts: {unique_counts[0]}")
    #print(f"First value in frequencies: {frequencies[0]}")
    print(f"last value in unique_counts: {unique_counts[-1]}")
    print(f"last value in frequencies: {frequencies[-1]}")

    return unique_counts, frequencies

def fit_gamma(data):
    shape, loc, scale = stats.gamma.fit(data, floc=0)  
    return shape, scale

def plot_gamma_qq(data, shape, scale):
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data, dist=stats.gamma, sparams=(shape, 0, scale), plot=ax)
    ax.get_lines()[1].set_color('red')
    ax.set_title('Q-Q Plot: K-mer Count vs. Gamma Distribution')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Observed Quantiles')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    output_path = "/data/Kaoutar/dev/py/kmers/figures/qqplot_gamma.png"
    unique_counts, frequencies = load_and_aggregate_counts(input_path, min_threshold=2, max_threshold=10000)
    shape, scale = fit_gamma(unique_counts)
    plot_gamma_qq(unique_counts, shape, scale)
