from torch.utils.data import DataLoader, Dataset
import pandas as pd
from musr.train_verifier import VERIFIER_QUESTION
from custom_datasets.verifier_dataset import VerifierDataset

class VerifierDataLoader(DataLoader):
    def __init__(self, df: pd.DataFrame, batch_size: int = 1, shuffle: bool = False):
        dataset = self.create_dataset(df)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

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
        
        
        
        
        
        
        
        