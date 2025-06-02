import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

file_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
output_png = "/data/Kaoutar/dev/py/kmers/figures/smotheted_histo"
output_pdf = "/data/Kaoutar/dev/py/kmers/figures/smotheted_histo"

df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["kmer_count", "num_kmers"])
print(df.head())  

counts = df["kmer_count"].values   
frequencies = df["num_kmers"].values  

sigma_value = 2.0
smoothed_frequencies = gaussian_filter1d(frequencies.astype(float), sigma=sigma_value)

plt.figure(figsize=(10, 5))
plt.plot(counts, frequencies, label="Original", color="steelblue", alpha=0.6)
plt.plot(counts, smoothed_frequencies, label=f"Smoothed (σ={sigma_value})", color="red", linewidth=2)

plt.xscale("log")  
#plt.xlim(1, 100) 
plt.title("Smoothed K-mer Frequency Distribution (Log Scale)")
plt.xlabel("K-mer Count (Occurrence Frequency) [Log Scale]")
plt.ylabel("Number of K-mers")
plt.legend()
plt.grid(True)
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, dpi=300, bbox_inches= "tight")
plt.show()
