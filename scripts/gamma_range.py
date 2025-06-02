import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_data_numpy(filename):
    data = np.loadtxt(filename)
    counts = data[:, 0].astype(int)
    frequencies = data[:, 1].astype(int)
    return counts, frequencies

def filter_counts_with_frequencies(counts, frequencies, lower=10, upper=10000):
    mask = (counts >= lower) & (counts <= upper)
    return counts[mask], frequencies[mask]

def expand_weighted_data(counts, frequencies):
    return np.repeat(counts, frequencies)

def estimate_gamma_params_weighted(counts, frequencies):
    expanded_data = expand_weighted_data(counts, frequencies)
    shape, loc, scale = stats.gamma.fit(expanded_data, floc=0)
    return shape, scale

def plot_qq_gamma(data, shape, scale, output_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data, dist=stats.gamma, sparams=(shape, 0, scale), plot=ax)
    ax.get_lines()[1].set_color('red')
    ax.set_title("QQ-Plot: Filtered K-mer Counts (10–10,000) vs Gamma Distribution")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
    plt.show()

def main(input_path, output_dir):
    counts, frequencies = load_data_numpy(input_path)
    filtered_counts, filtered_frequencies = filter_counts_with_frequencies(counts, frequencies, lower=10, upper=10000)
    expanded_counts = expand_weighted_data(filtered_counts, filtered_frequencies)
    shape, scale = estimate_gamma_params_weighted(filtered_counts, filtered_frequencies)
    output_path = f"{output_dir}/qqplot_gamma1.png"
    plot_qq_gamma(expanded_counts, shape, scale, output_path)
    print(f"Graphique enregistré dans {output_path}")

if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
    output_dir = "/data/Kaoutar/dev/py/kmers/figures"
    main(input_path, output_dir)
