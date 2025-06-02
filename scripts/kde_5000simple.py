import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
output_path = "/data/Kaoutar/dev/py/kmers/figures/all_data_fig/100000.fig"

def plot_kmer_distribution(df, save_path=None, sample_size=100000):
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df["Count"], color="blue", linewidth=2)
    
    plt.xlabel("K-mer Count", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("K-mer Count Distribution", fontsize=16)
    
    sns.despine()
    
    if save_path:
        plt.savefig(f"{save_path}/kmer_distribution.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}/kmer_distribution.pdf", dpi=300, bbox_inches="tight")

    plt.show()
df = pd.read_csv(input_file, sep=" ", names=["Kmer", "Count"])
plot_kmer_distribution(df, save_path=output_path)
