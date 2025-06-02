
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

histo_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"  
data = np.loadtxt(histo_file) 

kmer_counts = data[:, 0]  
num_kmers = data[:, 1]    

valid_range = (kmer_counts >= 1) & (kmer_counts <= 100)
kmer_counts_limited = kmer_counts[valid_range]
num_kmers_limited = num_kmers[valid_range]
smoothed_counts_limited = gaussian_filter1d(num_kmers_limited, sigma=2.0)
derivative_limited = np.gradient(smoothed_counts_limited)
local_min_index = np.where(np.diff(np.sign(derivative_limited)) > 0)[0][0]
optimal_threshold = kmer_counts_limited[local_min_index]
plt.figure(figsize=(8, 5))
plt.plot(kmer_counts, gaussian_filter1d(num_kmers, sigma=2.0), label="Smoothed K-mer Spectrum", color="red")
plt.axvline(x=optimal_threshold, color="green", linestyle="--", label=f"Threshold = {int(optimal_threshold)}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("K-mer Frequency (Log Scale)")
plt.ylabel("Number of K-mers")
plt.title("K-mer Frequency Distribution with Local Minimum")
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(kmer_counts_limited, smoothed_counts_limited, label="Smoothed K-mer Spectrum (1-100)", color="blue")
plt.axvline(x=optimal_threshold, color="green", linestyle="--", label=f"Corrected Threshold = {int(optimal_threshold)}")
plt.xlabel("K-mer Frequency")
plt.ylabel("Number of K-mers")
plt.title("Zoomed K-mer Spectrum (1-100)")
plt.legend()
plt.grid()
plt.show()

print(f"✔ Corrected K-mer Filtering Threshold: {optimal_threshold:.0f}")


