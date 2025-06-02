import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 
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

def read_fasta_file(file_path):
    sequences = []
    counts = []
    #n = 0
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
    #plt.show()

    
def process_kmers(input_file):
    output_dir = "/data/Kaoutar/dev/py/kmers/filtered_kmers"
    
    print(f"Reading sequences from {input_file}...")
    sequences, counts = read_fasta_file(input_file)

    df = pd.DataFrame({'Sequence': sequences, 'Count': counts})
    df = df[~((df["Sequence"] == "AAAAAAAAAAAAAAAAAAAAA") & (df["Count"] == 3774688))]
    print(df.head())  

    lambda_poisson = df["Count"].mean()
    std_poisson = np.sqrt(lambda_poisson)
    print(f"E = {lambda_poisson}, STD = {std_poisson}")
    #for cut in [1, 2, 3]:
        #print("cut", cut, "value", lambda_poisson + (std_poisson*cut) )
        #print("cut", cut, "value"  ,max(0, lambda_poisson - (std_poisson * cut)))

    #barplot(df["Count"].tolist())

    """cutoffs = {"cut1": 9, "cut2": 11, "cut3": 14}
    
    for name, cutoff in cutoffs.items():
        filtered_df = df[df["Count"] >= cutoff]
        output_file = f"{output_dir}/filtered_{name}.csv"
        print(output_file)
        #filtered_df.to_csv(output_file, index=False)
        #print(f"Filtered dataset saved at: {output_file} (≥{cutoff} counts, {len(filtered_df)} k-mers)")"""

if __name__ == "__main__":
    input_file = "/data/Kaoutar/dev/py/kmers/test_data/189N_lane1_20220816000_S16_L001_kmers_counts.txt"
    process_kmers(input_file)
#print(len(sequences))/data/Kaoutar/dev/py/kmers/test_data
    #print(len(counts))
    #print('sequence:')
    #print(sequences[-1])
    #print('counts:')
    #print(counts[-1])