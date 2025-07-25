from torch.utils.data import DataLoader, Dataset
import pandas as pd
from musr.train_verifier import VERIFIER_QUESTION
from custom_datasets.verifier_dataset import VerifierDataset

def custom_collate_fn(batch):
    case_names = [item['case_names'] for item in batch]
    mystery_texts = [item['mystery_texts'] for item in batch]
    suspects_lists = [item['suspects_lists'] for item in batch]
    true_labels = [item['true_labels'] for item in batch]
    generated_cots = [item['generated_cots'] for item in batch]
    texts = [item['text'] for item in batch]
    
    return {
        'case_names': case_names,
        'mystery_texts': mystery_texts,
        'suspects_lists': suspects_lists,
        'true_labels': true_labels,
        'generated_cots': generated_cots,
        'text': texts,
    }

class VerifierDataLoader(DataLoader):
    def __init__(self, df: pd.DataFrame, batch_size: int = 1, shuffle: bool = False):
        dataset = self.create_dataset(df)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)

    @staticmethod
    def create_dataset(df: pd.DataFrame):
        texts = []
        case_names = df['case_names'].tolist()
        mystery_texts = df['mystery_texts'].tolist()
        generated_cots = df['generated_cots'].tolist()
        suspects_lists = df['suspects_lists'].tolist()
        true_labels = df['true_labels'].tolist()
        
        for _, row in df.iterrows():
            text = (
                str(row["mystery_texts"])
                + "\nSuspects: " + str(row["suspects_lists"])
                + "\nQuestion: " + VERIFIER_QUESTION
                + "\nChain of Thought: " + str(row["generated_cots"])
            )
            texts.append(text)
        
        
        return VerifierDataset(texts, mystery_texts, suspects_lists, true_labels, case_names, generated_cots)
        
        
        
        
        
        
        
        