import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer


# input_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_31_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
# output_path = "/data/Kaoutar/dev/py/kmers/figures/plot_31_figs_100000"

def plot_kmer_distribution_kde(input_file:str, output_path:str, sample_size:int=10000):
    df = pd.read_csv(input_file, sep=" ", names=["Kmer", "Count"], engine="python")
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df["Count"], color="blue", linewidth=2, ax=ax)

    mean_kmer = df["Count"].mean()
    median_kmer = df["Count"].median()
 
    ax.axvline(x=mean_kmer, linestyle="-", linewidth=2, color="black", alpha=0.8)
    ax.text(mean_kmer, ax.get_ylim()[1] * 0.9, f"Mean = {int(mean_kmer)}",
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

if __name__ == "__main__":
    typer.run(plot_kmer_distribution_kde)
