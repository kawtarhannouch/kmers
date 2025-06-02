import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def load_and_filter_jellyfish(file_path: str):
    counts = np.loadtxt(file_path, dtype=int, usecols=1)  
    return counts[(counts >= 10) & (counts <= 10000)]  

def fit_negative_binomial(data: np.ndarray):
    mean, var = np.mean(data), np.var(data)
    print(f"Mean: {mean}, Variance: {var}")
    r = mean**2 / (var - mean) 
    p = mean / var  
    return r, p

def plot_negative_binomial_qq(data, output_dir):
    r, p = fit_negative_binomial(data)
    
    stats.probplot(data, dist=stats.nbinom, sparams=(r, p), plot=plt)
    
    plt.title('Q-Q Plot: K-mer Counts (10–10,000) vs. Negative Binomial Distribution')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Observed Quantiles')
    plt.grid(True)

    output_path = os.path.join(output_dir, 'negative_binomial_qq_plot.png')
    plt.savefig(output_path, dpi=300)  
    plt.show()   
    plt.close()

if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"  
    output_dir = "/data/Kaoutar/dev/py/kmers/figures"

    data = load_and_filter_jellyfish(input_path)
    plot_negative_binomial_qq(data, output_dir)

