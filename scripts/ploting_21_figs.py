import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
output_path = "/data/Kaoutar/dev/py/kmers/figures/100000_fig"

def plot_kmer_distribution_kde(df, save_path=None, sample_size=100000):
  
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df["Count"], color="blue", linewidth=2, ax=ax)

    mean_kmer = df["Count"].mean()
    print(mean_kmer)
    array_df=df["Count"].to_numpy()
    mean=np.mean(array_df)
    print(mean)
    median_kmer = df["Count"].median()
    ax.axvline(x=mean_kmer, linestyle="-", linewidth=2, color="black", alpha=0.8)
    ax.text(mean_kmer, ax.get_ylim()[1] * 0.9, f"Mean = {(mean_kmer)}",
            color="black", fontsize=14, fontweight="bold", ha="center")
 
    ax.axvline(x=median_kmer, linestyle="--", linewidth=2, color="green", alpha=0.8)
    ax.text(median_kmer, ax.get_ylim()[1] * 0.85, f"Median = {int(median_kmer)}",
            color="green", fontsize=14, fontweight="bold", ha="center")
    
    x_max = np.percentile(df["Count"], 99.5)
    ax.set_xlim(left=0, right=x_max)
    ax.set_xticks(np.arange(0, x_max + 5, 5))

    ax.set_xlabel("K-mer Count", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)  
    ax.set_title("K-mer Count Distribution", fontsize=18)
    sns.despine()

    if save_path:
        plt.savefig(f"{save_path}/kmer_distribution.png", dpi=300, bbox_inches="tight", format="png")
        plt.savefig(f"{save_path}/kmer_distribution.pdf", dpi=300, bbox_inches="tight", format="pdf")

    plt.show()

df = pd.read_csv(input_file, sep=" ", names=["Kmer", "Count"])
plot_kmer_distribution_kde(df, save_path=output_path)
