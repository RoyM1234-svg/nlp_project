from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

class VerifierDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }

class VerifierDataLoader(DataLoader):
    VERIFIER_QUESTION = """Who is the most likely murderer?"""

    
    def __init__(self, df: pd.DataFrame, batch_size: int = 1, shuffle: bool = False):
        dataset = self.create_dataset(df)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def create_dataset(self, df: pd.DataFrame):
        # Format the text the same way as in verifier training
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            text = (
                str(row["mystery_texts"])
                + "\nSuspects: " + str(row["suspects_lists"])
                + "\nQuestion: " + self.VERIFIER_QUESTION
                + "\nChain of Thought: " + str(row["generated_cots"])
            )
            texts.append(text)
            labels.append(int(row["true_labels"]))
        
        return VerifierDataset(texts, labels)
        
        
        
        
        
        
        
        