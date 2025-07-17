from datasets.detective_dataset import DetectiveDataset
from torch.utils.data import DataLoader
import pandas as pd

class DetectiveDataLoader(DataLoader):
    def __init__(self, df: pd.DataFrame, batch_size: int = 1, shuffle: bool = True):
        dataset = self.create_dataset(df)
        
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         collate_fn=self.collate_fn()
                        )

    @staticmethod
    def create_dataset(df: pd.DataFrame) -> DetectiveDataset:
        """Create dataset from DataFrame."""
        mystery_texts = df['mystery_text'].tolist()
        suspects_lists = [DetectiveDataLoader.parse_answer_options(opts) 
                         for opts in df['answer_options']]
        true_labels = [DetectiveDataLoader.parse_true_label(label) 
                      for label in df['answer']]
        case_names = df['case_name'].tolist()
        answer_options = df['answer_options'].tolist()
        
        return DetectiveDataset(mystery_texts, suspects_lists, true_labels)
    

    @staticmethod
    def parse_answer_options(answer_options_text: str) -> list[str]:
        options = answer_options_text.split(';')
        names = []
        for option in options:
            if ')' in option:
                name = option.split(')', 1)[1].strip()
                names.append(name) 
        return names

    @staticmethod
    def parse_true_label(true_label_text: str) -> str:
        if ')' in true_label_text:
            return true_label_text.split(')', 1)[1].strip()
        else:
            return true_label_text.strip()

    
    @staticmethod
    def collate_fn():
        """Custom collate function that creates prompts and tokenizes them."""
        def collate(batch):
            mystery_texts = [item['mystery_text'] for item in batch]
            suspects_lists = [item['suspects'] for item in batch]
            true_labels = [item['true_label'] for item in batch]
            case_names = [item['case_name'] for item in batch]
            answer_options = [item['answer_options'] for item in batch]
            indices = [item['index'] for item in batch]
            
            return {
                'mystery_texts': mystery_texts,
                'suspects_lists': suspects_lists,
                'case_names': case_names,
                'answer_options': answer_options,
                'true_labels': true_labels,
                'indices': indices,
            }
        
        return collate
