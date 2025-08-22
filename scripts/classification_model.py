import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(42)


encoding = {'A':0,'C':1,'G':2,'T':3}


patients = [
    ("AGCTATTTAAACCCCAACAATTAATTTTAAA", 4),
    ("CAGAAATGCATAAAATGAATCTTATAAGGAA", 2),
    ("ACGTGTGCTCTTCCGATCTCAGCTTACCCAC", 1),
    ("CTGGGGAAGTTGAGGCCTCAGTGAGCTGTAA", 1),
    ("AAATATGTAACTTTATATTTTTATTATTATT", 1),
    ("GTTATGTGTAATGTATTTAAAACCTTATTTA", 1),
    ("ATGGAAGGAGTGTGGAGAAGTGTGAAAAGAA", 1),
    ("CCTCTGGGCAGTCCACCCTGCCTGCCAAGGC", 36),
    ("GCCTTGGACAAGCCTTTTTGGGGAATTTGAA", 11),
    ("CTAAATGGTTGCAAATAGTAACAATTAACAG", 32),

]

reference = [
    ("TTAGTTCAGAAATCAACATTTTATAATGAAA", 1),
    ("AAGTCCAATTAAACCTCTTTCGTTTGTAAAT", 1),
    ("AAATGCAAATCAAAACCACAATACTATCATG", 1),
    ("AAATTGGGATTTTGTTCTGGGGAAGTGCTGA", 1),
    ("ATCAGCCAGGTGTGGTAGCTTGAGCCTGTAA", 1),
    ("AAAGAGTTGCTATAGTTTGAATGTTTGTCTC", 1),
    ("ATGAAATTAGCTTATGATTTTTGTTCCCATA", 1),
    ("AGCGTCACTCTGTCACCCAGGCTGGAGTGCA", 1),
    ("ATAAGTAGAATCTCTGCCCCTAAGACGGTAA", 1),
    ("GTGTTGTTTTTTCTATGTTATCCCCGATTCA", 1),

]

class KmerDataset(Dataset):
    def __init__(self, data, label, k=31):
        self.items = []
        for seq, cnt in data:
            assert len(seq)==k
            toks = torch.tensor([encoding[ch] for ch in seq], dtype=torch.long)    
            feat = torch.tensor([math.log1p(cnt)], dtype=torch.float32)            
            lab  = torch.tensor(int(label), dtype=torch.long)                    
            self.items.append((toks, feat, lab))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class KmerDistributer(nn.Module):
    def __init__(self, vocab=4, L=31, E=16, H=64, C=2, n_feat=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, E)
        self.fc1 = nn.Linear(L*E + n_feat, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.out = nn.Linear(H, C)
        self.act = nn.LeakyReLU(0.1)
        self.L, self.E = L, E
    def forward(self, tokens, feats):
        x = self.emb(tokens)          
        x = x.flatten(1)             
        x = torch.cat([feats, x], 1)  
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.out(x)         

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, tot, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for tokens, feats, labels in loader:
            tokens, feats, labels = tokens.to(device), feats.to(device), labels.to(device)
            if train: optimizer.zero_grad()
            logits = model(tokens, feats)            
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            preds = logits.argmax(1)                  
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            tot_loss += loss.item()*labels.size(0)
    return tot_loss/max(tot,1), correct/max(tot,1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds_pat = KmerDataset(patients, label=1)
    ds_ref = KmerDataset(reference, label=0)
    items  = ds_pat.items + ds_ref.items

    sequences = torch.stack([t for (t,_,_) in items])
    feats     = torch.stack([f for (_,f,_) in items])
    labels    = torch.stack([y for (_,_,y) in items])

 
    n = len(labels); n_train = max(2, int(0.8*n))
    idx = torch.randperm(n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    train_loader = DataLoader(list(zip(sequences[tr_idx], feats[tr_idx], labels[tr_idx])),
                              batch_size=8, shuffle=True)
    val_loader   = DataLoader(list(zip(sequences[va_idx], feats[va_idx], labels[va_idx])),
                              batch_size=8, shuffle=False)

   
    L = sequences.size(1)  
    model = KmerDistributer(vocab=4, L=L, E=16, H=64, C=2, n_feat=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

 
    for epoch in range(20):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:02d} | train loss {tr_loss:.3f} acc {tr_acc:.2f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.2f}")

  
    torch.save(model.state_dict(), "kmer_classifier_ce.pt")
    print("Saved:", "kmer_classifier_ce.pt")

    model.eval()
    with torch.no_grad():
        for tokens, feats, labels in val_loader:
            logits = model(tokens.to(device), feats.to(device))          
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()  
            for p, y in zip(probs, labels.tolist()):
                print(f"P(patient)={p:.3f} | true={y}")
            break

if __name__ == "__main__":
    main()
