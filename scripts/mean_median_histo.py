import numpy as np

histo_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
data = np.loadtxt(histo_file, dtype=int)  

kmer_counts = data[:, 0]  
num_kmers = data[:, 1]    
expanded_kmers_before = np.repeat(kmer_counts, num_kmers)  
median_kmer_before = np.median(expanded_kmers_before)
mean_kmer_before = np.sum(kmer_counts * num_kmers) / np.sum(num_kmers)

singleton_mask = kmer_counts > 1
kmer_counts_filtered = kmer_counts[singleton_mask]
num_kmers_filtered = num_kmers[singleton_mask]

expanded_kmers_after = np.repeat(kmer_counts_filtered, num_kmers_filtered)
median_kmer_after = np.median(expanded_kmers_after)
mean_kmer_after = np.sum(kmer_counts_filtered * num_kmers_filtered) / np.sum(num_kmers_filtered)

print(f"   - Median K-mer Frequency: {median_kmer_before:.0f}")
print(f"   - Mean K-mer Frequency: {mean_kmer_before:.2f}")

print(f"   - Median K-mer Frequency: {median_kmer_after:.0f}")
print(f"   - Mean K-mer Frequency: {mean_kmer_after:.2f}")
