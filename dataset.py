from torch.utils.data import DataLoader, Dataset


class TextDataLoader(Dataset):
    def __init__(self, text, summary, transpose=False):
        self.transpose = transpose
        self.text = text
        self.summary = summary
        
    def __getitem__(self, idx):
        if self.transpose:
            return self.text[idx].T, self.summary[idx].T
        else:
            return self.text[idx], self.summary[idx]
    
    def __len__(self): 
        return len(self.summary)