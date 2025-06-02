import numpy as np
import pandas as pd
import scipy.stats as stats 
import matplotlib.pyplot as plt
import os


def load_kmer_histogram(file_path):
    data = np.loadtxt(file_path, dtype=int)
    df = pd.DataFrame(data, columns=['count', 'frequency'])
    return df


def filter_and_expand_data(df, count_threshold=10):
    df_filtered = df[df['count'] < count_threshold]
    expanded_data = df_filtered.loc[df_filtered.index.repeat(df_filtered['frequency']), 'count']
    return expanded_data


def plot_poisson_qq(data, output_dir):
    stats.probplot(data, dist=stats.poisson(mu=data.mean()), plot=plt)
    plt.title('Q-Q plot: K-mer counts vs. Poisson distribution')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Observed quantiles')
    plt.grid(True)
    output_path = os.path.join(output_dir, 'poisson_qq_plot.png')
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    input_path = "/data/Kaoutar/dev/py/kmers/test_data/trimming/jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001.histo"
    output_dir = "/data/Kaoutar/dev/py/kmers/figures"

    df = load_kmer_histogram(input_path)
    expanded_data = filter_and_expand_data(df, count_threshold=10)
    plot_poisson_qq(expanded_data, output_dir)


