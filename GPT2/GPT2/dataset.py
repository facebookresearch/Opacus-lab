import torch
from torch.utils.data import Dataset, DataLoader

def load_wikitext(path):
    corpus = dict()
    for dset in ['valid', 'train', 'test']:
        corpus[dset] = torch.load(f'{path}/wikitext-103-{dset}-corpus.pt')
    return corpus

class CorpusDataset(Dataset):
    def __init__(self, corpus, seqlen):
        super().__init__()
        self.corpus = corpus
        self.seqlen = seqlen
    
    def __len__(self):
        return int(len(self.corpus)/self.seqlen)

    def __getitem__(self, item):
        idx = item*self.seqlen
        return self.corpus[idx:idx+self.seqlen]

    
class UserLvlDataset(Dataset):
    def __init__(self, users):
        super().__init__()
        #a list of users
        self.users = users #each user is a CorpusDataset

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        idx = torch.randint(len(self.users[item]),(1,)).item()
        return self.users[item][idx]
