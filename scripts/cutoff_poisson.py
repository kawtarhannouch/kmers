import numpy as np
import math
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
import pandas as pd 
import numpy as np 
import random
class Encoder:
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoding = {
            'A': 0,
            'T': 1,
            'C': 2,
            'G': 3,
        }
        self.decoding = None
        self._make_decoding()

    def _make_decoding(self):
        self.decoding = {value: key for key, value in self.encoding.items()}

    def encode(self, char):
        return self.encoding[char]

    def split_seq(self, sequence):
        sub_seq = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
        return sub_seq

    def decode(self, val):
        return self.decoding[val]

    def encode_sequence(self, sequence):
        split_sequence = self.split_seq(sequence)
        encoded_pairs = []
        for seq in split_sequence:
            encoded = 0
            n = 0
            for char in seq:
                encoded += 2 ** (self.encode(char) + n)
                n += 4
            encoded_pairs.append(encoded)
        return encoded_pairs

    def decode_sequence(self, encoded_pairs):
        decoded_sequence = []
        for encoded in encoded_pairs:
            pair = []
            while encoded > 0:
                n = int(math.log2(encoded))
                char = self.decode(n % 4)
                pair.append(char)
                encoded -= 2 ** n
            decoded_sequence.append(''.join(pair[::-1]))
        return ''.join(decoded_sequence)

def test_encode_decode(sequence):
    encoder = Encoder()
    enc = encoder.encode_sequence(sequence)
    dec = encoder.decode_sequence(enc)
    res = sequence == dec
    return res, enc, dec

def read_fasta_file(file_path, nb_sequence):
    sequences = []
    counts = []
    n = 0
    with open(file_path, 'r') as f:
        
        for i, line in enumerate(f):
            line = line.strip()
            
            if line.startswith('>'):
                count = int(line[1:])
                counts.append(count)
                
            elif line[0] in ['A', 'T', 'G', 'C']:
                sequences.append(line)
            else:
                raise ValueError(f'Faulty line {i}')

            if n >= nb_sequence-1:
                break
            n += 1
    
    return sequences, counts

def save_file(file_name, sequences, counts):
    with open(file_name, 'w') as file:
        for seq, count in zip(sequences, counts):
            file.write(f"{count}\n{seq}\n")

def save_encoded_sequences_as_numpy(filename, encoded_sequences):
    np_array = np.array(encoded_sequences, dtype=np.uint8)
    np.save(filename, np_array)

def compare_sequence(original_file, decoded_file):
    with open(original_file, 'r') as f1, open(decoded_file, 'r') as f2:
        original = f1.readlines()
        decoded = f2.readlines()
        return original == decoded

def barplot(counts):
    
    sampled_counts = random.sample(counts, min(5000, len(counts)))
    df = pd.DataFrame({'counts': sampled_counts})
    rslt = df['counts'].value_counts().reset_index()
    rslt.columns = ['counts', 'frequency']
    rslt.sort_values(by='counts', inplace=True)

    #print(rslt['counts'].tolist())
    #print(rslt['frequency'].tolist())

    plt.figure(figsize=(20, 10))
    plt.bar(rslt['counts'].astype(str), rslt['frequency'])
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.title('Distribution of K-mer Counts')

    tick_positions = range(0, len(rslt), max(1, len(rslt) // 20))
    plt.xticks(tick_positions, rslt['counts'].astype(str).iloc[tick_positions], rotation=90)
    plt.show()

"""def barplot(counts):
    df = pd.DataFrame({'counts': counts})
    rslt = df['counts'].value_counts().reset_index()
    rslt.columns = ['counts', 'frequency']
    print(rslt['counts'].tolist())
    print(rslt['frequency'].tolist())
    rslt.sort_values(by='counts', inplace=True)

    plt.figure(figsize=(20, 14))
    plt.bar(rslt['counts'].astype(str), rslt['frequency'])
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.title('Distribution of K-mer Counts')

    tick_positions = range(0, len(rslt), 5) 
    plt.xticks(tick_positions, rslt['counts'].astype(str).iloc[tick_positions], rotation=90)

    plt.show() """


def plot_kmer_distribution(df, cutoffs, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.histplot(df["Count"], bins=50, kde=True, color="blue", alpha=0.7, ax=ax, edgecolor="black")

    for name, cutoff in cutoffs.items():
        if "plus" in name:
            ax.axvline(x=cutoff, linestyle="--", linewidth=2, color="red", alpha=0.8, label=f"{name} ({cutoff})")

    x_max = np.percentile(df["Count"], 99.5)
    ax.set_xlim(left=0, right=x_max)
    ax.set_xticks(np.arange(0, x_max + 5, 5))

    ax.set_xlabel("K-mer Count", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title("K-mer Count Distribution", fontsize=18)
    ax.legend(loc="upper right", frameon=True, fontsize=14)
    sns.despine()

    
    if save_path:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight", format="png")  
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight", format="pdf")  
    
    plt.show()

def plot_kmer_distribution_cutoffs(df, cutoffs, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df["Count"], bins=1000, kde=True, color="blue", alpha=0.7, ax=ax, edgecolor="black")

    mean_kmer = df["Count"].mean()
    ax.axvline(x=mean_kmer, linestyle="-", linewidth=2, color="black", alpha=0.8)
    ax.text(mean_kmer, ax.get_ylim()[1] * 0.9, f"Mean = {int(mean_kmer)}",
            color="black", fontsize=14, fontweight="bold", ha="center")
    for name, cutoff in cutoffs.items():
        if "plus" in name:  
            ax.axvline(x=cutoff, linestyle="--", linewidth=2, color="red", alpha=0.8)
            ax.text(cutoff, ax.get_ylim()[1] * 0.85, f"{int(cutoff)}",
                    color="red", fontsize=14, fontweight="bold", ha="center")

    x_max = np.percentile(df["Count"], 99.5)
    ax.set_xlim(left=0, right=x_max)
    ax.set_xticks(np.arange(0, x_max + 5, 5))

    ax.set_xlabel("K-mer Count", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_title("K-mer Count Distribution", fontsize=18)
    sns.despine()
    if save_path:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight", format="png")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight", format="pdf")

    plt.show()

def plot_kmer_distribution_kde(df, cutoffs, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df["Count"], color="blue", linewidth=2, ax=ax)

    mean_kmer = df["Count"].mean()
    ax.axvline(x=mean_kmer, linestyle="-", linewidth=2, color="black", alpha=0.8)
    ax.text(mean_kmer, ax.get_ylim()[1] * 0.9, f"Mean = {int(mean_kmer)}",
            color="black", fontsize=14, fontweight="bold", ha="center")

    for name, cutoff in cutoffs.items():
        if "plus" in name:
            ax.axvline(x=cutoff, linestyle="--", linewidth=2, color="red", alpha=0.8)
            ax.text(cutoff, ax.get_ylim()[1] * 0.85, f"{int(cutoff)}",
                    color="red", fontsize=14, fontweight="bold", ha="center")

    x_max = np.percentile(df["Count"], 99.5)
    ax.set_xlim(left=0, right=x_max)
    ax.set_xticks(np.arange(0, x_max + 5, 5))

    ax.set_xlabel("K-mer Count", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)  
    ax.set_title("K-mer Count Distribution", fontsize=18)
    sns.despine()

    if save_path:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight", format="png")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight", format="pdf")

    plt.show()

def plot_kmer_distribution_aftercutof( df, cutoff_value, mean_value,name_fig):
    #figures_dir = "/data/Kaoutar/dev/py/kmers/figures/"
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Count"], bins=1000, color="blue", alpha=0.6)
    plt.axvline(cutoff_value, color='black', linestyle='dashed', linewidth=2)
    plt.axvline(mean_value, color='green', linestyle='solid', linewidth=2)
    plt.xlabel("K-mer Count", fontsize=16, fontweight='bold')
    plt.ylabel("Frequency", fontsize=16, fontweight='bold')
    plt.title(f"K-mer Count Distribution - {cutoff_value}", fontsize=18, fontweight='bold')
    sns.despine()
    plt.savefig(name_fig + "_.pdf", dpi=300, bbox_inches="tight", format="pdf")
    #plt.show()

def plot_kmer_distribution_aftercutoff(df, cutoff_value, mean_value, name_fig): 
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    
    plt.figure(figsize=(10, 6))  
    plt.hist(df['Count'], bins=400, color="blue", alpha=0.6, log=True)
    for value, color, linestyle, label in [
        (cutoff_value, 'red', 'dashed', f'Cutoff: {cutoff_value}'),
        (mean_value, 'green', 'solid', f'Mean: {mean_value:.2f}')
    ]:
        plt.axvline(value, color=color, linestyle=linestyle, linewidth=2, label=label)

    plt.xscale('log')
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))  
    plt.xlabel("K-mer Count (Log Scale)", fontsize=16, fontweight='bold')
    plt.ylabel("Log-scaled Frequency of K-mer Counts", fontsize=16, fontweight='bold')
    plt.title(f"K-mer Count Distribution (Cutoff = {cutoff_value})", fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, which="major", linestyle="--", linewidth=0.7)
    plt.grid(False, which="minor")
    sns.despine()
    for fmt in ["pdf", "png"]:
        plt.savefig(f"{name_fig}.{fmt}", dpi=300, bbox_inches="tight", format=fmt)

    plt.show()

if __name__ == '__main__':
    sequences, counts = read_fasta_file("/data/Kaoutar/dev/py/kmers/test_data/189N_lane1_20220816000_S16_L001_kmers_counts.txt", 100000)
    df = pd.DataFrame({'Sequence': sequences, 'Count': counts})
    #df = df[~((df["Sequence"] == "AAAAAAAAAAAAAAAAAAAAA") & (df["Count"] == 3774688))]
    output_dir = "/data/Kaoutar/dev/py/kmers/filtered_kmers/cutoffs_cdf"
    lambda_poisson = df["Count"].mean()
    std_poisson = np.sqrt(lambda_poisson)
    print(f"E = {lambda_poisson}, STD = {std_poisson}")
    #min= lambda_poisson-3 * std_poisson
    #max= lambda_poisson+3 * std_poisson
    
    cumulative_prob = 0.05
    f_min = poisson.ppf(cumulative_prob, lambda_poisson)
    filtered_df = df[df['Count'] > f_min]
    print(f_min)
    upper_cumulative_prob = 0.95
    f_max = poisson.ppf(upper_cumulative_prob, lambda_poisson)
    print(f_max)

    filtered_df = df[(df['Count'] > f_min) & (df['Count'] < f_max)]
    print(filtered_df.head())

    output_file = "/data/Kaoutar/dev/py/kmers/filtered_kmers/cutoffs_cdf/poisson_filtered.csv"
    filtered_df.to_csv(output_file, index=False)

