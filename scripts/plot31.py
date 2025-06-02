import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_kmer_distribution(input_file, nrows=None, save_prefix=None):
    
    df = pd.read_csv(input_file, sep=' ', header=None, names=['Kmer', 'Count'], nrows=nrows)
    df_filtered = df[(df["Count"] > 1) & (df["Kmer"] != "AAAAAAAAAAAAAAAAAAAAA")]
    
    median_kmer = df_filtered["Count"].median()
    mean_kmer = df_filtered["Count"].mean()
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    ax_hist.hist(df_filtered["Count"], bins=200)

    ax_hist.set_xlabel("K-mer Count", fontsize=16)
    ax_hist.set_ylabel("Frequency", fontsize=16)
    ax_hist.set_title("K-mer Count Distribution (Histogram)", fontsize=18)
    
    x_max = np.percentile(df_filtered["Count"], 99.5)
    ax_hist.set_xlim(0, x_max)
    ax_hist.axvline(x=median_kmer, linestyle='-', linewidth=2, color="black", alpha=0.8)
    ax_hist.text(
        median_kmer + (0.02 * x_max),
        ax_hist.get_ylim()[1]*0.9,
        f"Median = {median_kmer:.2f}",
        color="black",
        fontsize=14,
        fontweight="bold",
        ha="left"
    )
    ax_hist.axvline(x=mean_kmer, linestyle='--', linewidth=2, color="red", alpha=0.8)
    ax_hist.text(
        mean_kmer + (0.02 * x_max),
        ax_hist.get_ylim()[1]*0.8,
        f"Mean = {mean_kmer:.2f}",
        color="red",
        fontsize=14,
        fontweight="bold",
        ha="left"
    )

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_hist.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_prefix}_hist.pdf", dpi=300, bbox_inches="tight")

    plt.show()

    fig_kde, ax_kde = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df_filtered["Count"], color="blue", linewidth=2, bw_adjust=0.5, ax=ax_kde)

    ax_kde.axvline(x=median_kmer, linestyle='-', linewidth=2, color="black", alpha=0.8)
    ax_kde.text(
        median_kmer + (0.02 * x_max),
        ax_kde.get_ylim()[1]*0.9,
        f"Median = {median_kmer:.2f}",
        color="black",
        fontsize=14,
        fontweight="bold",
        ha="left"
    )
    ax_kde.axvline(x=mean_kmer, linestyle='--', linewidth=2, color="red", alpha=0.8)
    ax_kde.text(
        mean_kmer + (0.02 * x_max),
        ax_kde.get_ylim()[1]*0.8,
        f"Mean = {mean_kmer:.2f}",
        color="red",
        fontsize=14,
        fontweight="bold",
        ha="left"
    )

    ax_kde.set_xlim(0, x_max)
    ax_kde.set_xlabel("K-mer Count", fontsize=16)
    ax_kde.set_ylabel("Density", fontsize=16)
    ax_kde.set_title("K-mer Count Distribution (KDE)", fontsize=18)
    
    sns.despine()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_kde.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_prefix}_kde.pdf", dpi=300, bbox_inches="tight")

    plt.show()

input_file = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_31_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
output_prefix = "/data/Kaoutar/dev/py/kmers/figures/kmer_distribution"

plot_kmer_distribution(input_file, nrows=None, save_prefix=output_prefix)

