from torch.utils.data import Dataset
from typing import List

class DetectiveDataset(Dataset):
    """PyTorch Dataset for detective puzzles."""
    
    def __init__(self, 
                 mystery_texts: List[str], 
                 suspects_lists: List[List[str]],
                 true_labels: List[str],
                 case_names: List[str]):
        self.mystery_texts = mystery_texts
        self.suspects_lists = suspects_lists
        self.true_labels = true_labels
        self.case_names = case_names
        
    def __len__(self):
        return len(self.mystery_texts)
    
    def __getitem__(self, idx):
        return {
            'mystery_text': self.mystery_texts[idx],
            'suspects': self.suspects_lists[idx],
            'true_label': self.true_labels[idx],
            'case_name': self.case_names[idx],
            'index': idx
        }
    

