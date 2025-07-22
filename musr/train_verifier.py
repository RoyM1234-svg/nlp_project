from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorWithPadding
import pandas as pd
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
import wandb
import os
import numpy as np
from transformers.trainer_utils import EvalPrediction
from transformers.trainer import Trainer



MAX_LEN = 512

@dataclass
class AdditionalTrainingArguments:
    lr: float = field(metadata={"help": "Learning rate for training."})
    batch_size: int = field(default=16, metadata={"help": "Batch size for training."})
    save_model_path: str = field(default="verifier_model", metadata={"help": "Path to save the model."})
    model_name: str = field(default="microsoft/deberta-v3-base", metadata={"help": "Name of the model to use."})


def preprocess_data(df: pd.DataFrame) -> DatasetDict:
    df["text"] = (
        df["story"]
        + "\nSuspects: " + df["suspects"]
        + "\nQuestion: " + df["question"]
        + "\nChain of Thought: " + df["flat_chain"]
    )

    dataset = Dataset.from_pandas(df[["text", "label"]]).train_test_split(test_size=0.1,seed=42)

    return dataset

def read_csv(file_name: str) -> pd.DataFrame:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(current_dir, file_name))

def compute_metrics(pred: EvalPrediction):
    logits, labels = pred
    probs = 1 / (1 + (1 / np.exp(logits))) # sigmoid in NumPy
    preds = (probs > 0.5).astype(np.int32)
    acc   = (preds == labels).mean().item()
    return {"accuracy": acc}

def train_verifier(training_args: TrainingArguments, additional_args: AdditionalTrainingArguments):
    tokenizer = AutoTokenizer.from_pretrained(additional_args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        additional_args.model_name,
        num_labels=1,
        problem_type = "single_label_classification"
    )

    df = read_csv("verifier_training_data/MUSR_data.csv")

    dataset = preprocess_data(df)

    print(dataset)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch",
                   columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


def main():

    parser = HfArgumentParser((TrainingArguments, AdditionalTrainingArguments)) # type: ignore
    training_args, additional_args = parser.parse_args_into_dataclasses()

    training_args.per_device_train_batch_size = additional_args.batch_size
    training_args.per_device_eval_batch_size = additional_args.batch_size
    training_args.learning_rate = additional_args.lr
    training_args.output_dir = additional_args.save_model_path
    training_args.report_to = "wandb"
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 1

    wandb.init(project="verifier_training",
               name=f"batch_size_{additional_args.batch_size}_lr_{additional_args.lr}",
               config = {
                "batch_size": additional_args.batch_size,
                "lr": additional_args.lr,
                "model_name": additional_args.model_name,
               })


    train_verifier(training_args, additional_args)

    wandb.finish()


if __name__ == "__main__":
    main()