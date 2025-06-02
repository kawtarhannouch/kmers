import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_kmer_distribution_kde(input_file, save_path=None):
    df = pd.read_csv(input_file, sep=' ', header=None, names=['Kmer', 'Count'], nrows=10000)
    df_filtered = df[df["Kmer"] != "AAAAAAAAAAAAAAAAAAAAA"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df_filtered["Count"], color="blue", linewidth=2, bw_adjust=0.5, ax=ax)
    median_kmer = df_filtered["Count"].median()
    ax.axvline(median_kmer, linestyle="-", linewidth=2, color="black", alpha=0.8)
    ax.text(median_kmer + 5, ax.get_ylim()[1] * 0.9, f"Median = {int(median_kmer)}", color="black", fontsize=14, fontweight="bold", ha="left")
    mean_kmer = df_filtered["Count"].mean()
    ax.axvline(mean_kmer, linestyle="-", linewidth=2, color="red", alpha=0.8)
    ax.text(mean_kmer + 5, ax.get_ylim()[1] * 0.8, f"Mean = {int(mean_kmer)}", color="red", fontsize=14, fontweight="bold", ha="left")
    x_max = np.percentile(df_filtered["Count"], 99.5)
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.round(np.linspace(0, x_max, 8), 1))
    ax.set_xlabel("K-mer Count", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title("K-mer Count Distribution", fontsize=18)
    sns.despine()
    if save_path:
        for ext in ["png", "pdf"]:
            plt.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches="tight", format=ext)
    plt.show()

input_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_31_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
output_prefix = "/data/Kaoutar/dev/py/kmers/figures/plot_31_figs/kmer_distribution"

plot_kmer_distribution_kde(input_file, save_path=output_prefix)