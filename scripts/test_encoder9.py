import numpy as np
import math
import matplotlib.pyplot as plt

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
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                count = int(line[1:])
                counts.append(count)
            else:
                sequences.append(line)
            if n >= nb_sequence:
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


def barplot(x,y):
    plt.bar(x,y)
    #plt.xscale('log')
    plt.xlabel('Nombre d\'occurrences des k-mers')
    plt.ylabel('Fréquence')
    plt.title('Distribution des occurrences des k-mers')
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    sequences, counts = read_fasta_file("/data/189N_lane1_20220816000_S16_L001_kmers_counts.txt", 5000)
    data = {'counts': counts}
    df = pd.DataFrame(data)
    rslt=df["counts"].value_counts()
    rslt=rslt.to_frame()
    rslt['counts'] = rslt.index
    rslt.reset_index(drop=True, inplace=True)
    rslt.columns = ['frequency', 'counts']
    x=rslt["counts"].tolist()
    y=rslt["frequency"].tolist()
    #barplot(x,y)
    ax=rslt.plot.bar(x='counts',y='frequency',rot=0)
    ax.show()5
  

    #save_file("original_sequence5000.txt", sequences, counts)

    """encoder = Encoder()
    enc_seq = []
    for seq in sequences:
        enc = encoder.encode_sequence(seq)
        enc_seq.append(enc)

    save_file("encoded_seq_5000.txt", enc_seq)
    save_encoded_sequences_as_numpy("encoded_seq_5000.npy", enc_seq)

    # Decoding (optional)
    decoded_sequences = []
    for encoded in enc_seq:
        dec = encoder.decode_sequence(encoded)
        decoded_sequences.append(dec)

    save_file("decoded_seq_5000.txt", decoded_sequences)

    original_file = "/data/Kaoutar/dev/py/original_sequence5000.txt"
    decoded_file = "/data/Kaoutar/dev/py/decoded_seq_5000.txt"
    comparaison_results = compare_sequence(original_file, decoded_file)
    if comparaison_results:
        print("original_file and decoded_file match.")
    else:
        print("original_file and decoded_file do not match.")"""

    """plot_counts(counts)
    plt.savefig('histogram.pdf')"""

    
