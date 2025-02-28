import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer,TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch.utils.data import Dataset
from transformers.trainer_utils import SchedulerType

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    if args.num_labels == 2: 
        f1 = f1_score(labels, predictions, average="binary")
    else:
        f1 = f1_score(labels, predictions, average="macro")

    return {
        "Accuracy": accuracy,
        "F1": f1,
        "MCC": mcc,
        }

def main(args):
    train_df = pd.read_csv(args.train_data_path)
    val_df = pd.read_csv(args.val_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = DataFrameDataset(train_df, tokenizer, args.max_length)
    val_dataset = DataFrameDataset(val_df, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        logging_dir=args.logging_dir,
        logging_strategy="epoch",
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_ratio=args.warmup_ratio,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clf model")
    parser.add_argument('--model_name', type=str, default='cointegrated/rubert-tiny2', help='Pretrained any model name for sequence classification.')
    parser.add_argument( "--train_data_path", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data CSV file.")
    parser.add_argument( "--num_labels", type=int, default=2, help="Number of labels for classification.")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the model and logs.")
    parser.add_argument("--logging_dir",type=str, default="./logs", help="Directory to save the training logs.")
    parser.add_argument('--warmup_ration', type=float, default=0.1, help='Number of warmup steps for scheduler')

    args = parser.parse_args()
    main(args)