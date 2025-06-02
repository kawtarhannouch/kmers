import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(filename) 
    return data[:, 0], data[:, 1]  

def filter_counts(count, threshold= 10):
    return count[count >= threshold]

def estimate_gamma_params(data):
    shape, loc, scale = stats.gamma.fit(data, floc=0)
    return shape, scale

def plot_qq_gamma(data, shape, scale, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data, dist=stats.gamma, sparams=(shape, 0, scale), plot=ax)
    ax.get_lines()[1].set_color('red')
    ax.set_title("QQ-Plot: K-mer Frequency vs Gamma Distribution")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

def main(input_path, output_dir):
    count, frequency = load_data(input_path)
    filtered_counts = filter_counts(count)
    shape, scale = estimate_gamma_params(filtered_counts)
    output_path = f"{output_dir}/qqplot_gamma1.png"
    plot_qq_gamma(filtered_counts, shape, scale, output_path)

if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    output_dir = "/data/Kaoutar/dev/py/kmers/figures"
    main(input_path, output_dir)

