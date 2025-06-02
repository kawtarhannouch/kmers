import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import mode

histogram_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"

def load_kmer_histogram(file_path):
    kmer_histogram = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                kmer, count = int(parts[0]), int(parts[1])
                kmer_histogram[kmer] = count
    return kmer_histogram

kmer_histogram = load_kmer_histogram(histogram_file)

kmer_values = np.array(list(kmer_histogram.keys()))
kmer_frequencies = np.array(list(kmer_histogram.values()))
def empirical_cutoff(kmer_values, kmer_frequencies):
    weighted_mode = kmer_values[np.argmax(kmer_frequencies)]  
    weighted_mean = np.average(kmer_values, weights=kmer_frequencies)  
    weighted_var = np.average((kmer_values - weighted_mean) ** 2, weights=kmer_frequencies)  
    weighted_std = np.sqrt(weighted_var)  
    
    t_cutoff = weighted_mode + 1.5 * weighted_std
    return int(np.round(t_cutoff))

t_cutoff = empirical_cutoff(kmer_values, kmer_frequencies)
df_kmers = pd.DataFrame({'K-mer Count': kmer_values, 'Frequency': kmer_frequencies})
summary_stats = df_kmers.describe()
print(summary_stats)
plt.figure(figsize=(10, 6))
bins = np.logspace(np.log10(1), np.log10(max(kmer_values)), 50) 
plt.bar(kmer_values, kmer_frequencies / np.sum(kmer_frequencies), width=0.8, alpha=0.7, color="royalblue", label="Observed k-mer Counts")
plt.axvline(t_cutoff, color="red", linestyle="dashed", linewidth=2, label=f"Cutoff: {t_cutoff}")

plt.xscale("log")  
plt.yscale("log")  
plt.xlabel("K-mer Count", fontsize=14, fontweight="bold")
plt.ylabel("Density", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("K-mer Count Distribution with Empirical Cutoff", fontsize=16, fontweight="bold")
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)
plt.savefig("kmer_distribution.png", dpi=300, bbox_inches="tight")  
plt.show()
t_cutoff

