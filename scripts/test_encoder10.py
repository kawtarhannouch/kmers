import numpy as np
import math
import typer
import random

class Encoder:
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoding = {
            'A' : 0,
            'T' : 1,
            'C' : 2,
            'G' : 3,
        }
        self.decoding = None
        self._make_decoding()

    def _make_decoding(self):
         self.decoding = {value: key for key, value in self.encoding.items()}

    def encode(self, char):
        return self.encoding[char]

    def split_seq(self, sequence):
       sub_seq = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
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
                encoded += 2**(self.encode(char) + n)
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
    dec =  encoder.decode_sequence(enc)
    res = sequence == dec
    return res, enc, dec

def read_fasta_file(file_path, nb_sequence):
    sequences = []
    n = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()        
            if not line.startswith('>'):   
                sequences.append(line)
            if n > nb_sequence + 1:
                break
            n += 1
    return sequences

def save_file(file_name, list_elements):
    with open(file_name, 'w') as file:
        for elt in list_elements:
            file.write(f"{elt}\n")

def save_encoded_sequences_as_numpy(filename, encoded_sequences):
    np_array = np.array(encoded_sequences, dtype=np.uint8)
    np.save(filename, np_array)

def compare_sequence(original_file, decoded_file):
    with open(original_file, 'r') as f1, open(decoded_file, 'r') as f2:
        original = f1.readlines()
        decoded = f2.readlines()
        return original == decoded

def plot_counts(counts):
    plt.hist(counts, bins=range(min(counts), max(counts) + 1), edgecolor='black')
    plt.xscale('log')
    plt.xlabel('Nombre d\'occurrences des k-mers')
    plt.ylabel('Fréquence')
    plt.title('Distribution des occurrences des k-mers')
    plt.show()


def main(fasta_file: str, output_file: str, nb_sequence: int = 5000):
    list_seq = read_fasta_file(fasta_file, nb_sequence)
    save_file(output_file, list_seq)
    
    enc_seq = []
    encoder = Encoder()
    for seq in list_seq:
        enc = encoder.encode_sequence(seq)
        enc_seq.append(enc)

    save_file("encoded_seq_5000.txt", enc_seq)
    
    decoded_sequences = []  
    for encoded in enc_seq:
        dec = encoder.decode_sequence(encoded) 
        decoded_sequences.append(dec)

    save_file("decoded_seq_5000.txt", decoded_sequences)

    comparaison_results = compare_sequence("original_sequence5000.txt", "decoded_seq_5000.txt")
    if comparaison_results:
        print("original_file and decoded_file match.")
    else:
        print("original_file and decoded_file do not match.")

if __name__ == "__main__":
   typer.run(main)

