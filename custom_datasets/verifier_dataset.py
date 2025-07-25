from torch.utils.data import Dataset

class VerifierDataset(Dataset):
    def __init__(
            self,
            texts,
            mystery_texts,
            suspects_lists,
            true_labels,
            case_names,
            generated_cots):
        
        self.texts = texts
        self.mystery_texts = mystery_texts
        self.suspects_lists = suspects_lists
        self.true_labels = true_labels
        self.case_names = case_names
        self.generated_cots = generated_cots
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'case_names': self.case_names[idx],
            'mystery_texts': self.mystery_texts[idx],
            'suspects_lists': self.suspects_lists[idx],
            'true_labels': self.true_labels[idx],
            'generated_cots': self.generated_cots[idx],
            'text': self.texts[idx],
        }
    