import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def load_and_filter_jellyfish(file_path: str, threshold: int = 10000):
    counts = np.loadtxt(file_path, dtype=int, usecols=1)  
    return counts[counts >= threshold]  

def fit_poisson(data: np.ndarray):
    lambda_poisson = np.mean(data)  
    return lambda_poisson

def plot_poisson_qq(data):
    lambda_poisson = fit_poisson(data)
    stats.probplot(data, dist=stats.poisson, sparams=(lambda_poisson,), plot=plt)
    
    plt.title('Q-Q Plot: K-mer Counts (>10,000) vs. Poisson Distribution')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Observed Quantiles')
    plt.grid(True)

    plt.show()  

if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"  

    data = load_and_filter_jellyfish(input_path, threshold=10000)
    plot_poisson_qq(data)
