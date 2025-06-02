import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

histo_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
data = np.loadtxt(histo_file, dtype=int)

kmer_counts = data[:, 0]  
print(kmer_counts)
num_kmers = data[:, 1]
print(num_kmers)    

filtered_indices = kmer_counts > 1  
kmer_counts_filtered = kmer_counts[filtered_indices]
print(kmer_counts_filtered)
num_kmers_filtered = num_kmers[filtered_indices]
print(num_kmers_filtered)

smoothed_counts_filtered = gaussian_filter1d(num_kmers_filtered, sigma=1.0)
derivative_filtered = np.gradient(smoothed_counts_filtered)
#print(derivative_filtered)
local_min_indexes_filtered = np.where(np.diff(np.sign(derivative_filtered)) > 0)[0]
#print(local_min_indexes_filtered)

if len(local_min_indexes_filtered) > 0:
    local_min_index_filtered = local_min_indexes_filtered[0]  
    optimal_threshold_filtered = kmer_counts_filtered[local_min_index_filtered]
else:
    print("⚠ No clear local minimum found after filtering, selecting the lowest dip.")
    local_min_index_filtered = np.argmin(smoothed_counts_filtered) 
    optimal_threshold_filtered = kmer_counts_filtered[local_min_index_filtered]

plt.figure(figsize=(8, 5))
plt.plot(kmer_counts_filtered, smoothed_counts_filtered, label="Smoothed K-mer Spectrum", color="red")
plt.axvline(x=optimal_threshold_filtered, color="green", linestyle="--", label=f"Corrected Threshold = {int(optimal_threshold_filtered)}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("K-mer Frequency (Log Scale)")
plt.ylabel("Number of K-mers")
plt.title("K-mer Frequency Distribution After Removing Singletons (Corrected)")
plt.legend()
plt.grid()
plt.show()


print(f"✔ Corrected K-mer Filtering Threshold: {optimal_threshold_filtered:.0f}")




















