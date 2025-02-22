import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Dataset for multi-task 
class TextDataset(Dataset):
    def __init__(self, texts, binary_labels, error_labels, tokenizer, max_length):
        self.texts = texts
        self.binary_labels = binary_labels
        self.error_labels = error_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'binary_label': torch.tensor(self.binary_labels[idx], dtype=torch.float),
            'error_label': torch.tensor(self.error_labels[idx], dtype=torch.long)
        }

def get_pooling(outputs, attention_masks, pooling_name="cls", hidden_states=1):
    last_hidden_state = outputs.hidden_states[-1]
    if hidden_states > 1:
      last_hidden_state = torch.cat(tuple([outputs.hidden_states[-i] for i in range(hidden_states, 0, -1)]), dim=-1)

    input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
    if pooling_name == "mean":
        mean_pooling = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return mean_pooling
    elif pooling_name == "cls":
        cls_pooling = last_hidden_state[:, 0, :]
        cls_pooling = torch.nn.functional.normalize(cls_pooling)
        return cls_pooling
    elif pooling_name == "max":
       max_pooling, _ = torch.max(last_hidden_state * input_mask_expanded, dim=1)
       return max_pooling

# Model for multi-task learning
class ModelMTL(nn.Module):
    def __init__(self, model_name, num_error_types, num_hidden_states=1, pooling_name="cls"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling_name = pooling_name
        self.num_hidden_states = num_hidden_states
        self.binary_classifier = nn.Linear(self.num_hidden_states * self.model.config.hidden_size, 1)
        self.error_classifier = nn.Linear(self.num_hidden_states * self.model.config.hidden_size, num_error_types)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled_output = get_pooling(outputs, attention_mask, pooling_name=self.pooling_name, hidden_states=self.num_hidden_states)
        pooled_output = self.dropout(pooled_output)
        
        # out for binary clf
        binary_logits = self.binary_classifier(pooled_output).squeeze(-1)
        # out for multi-class clf
        error_logits = self.error_classifier(pooled_output)
        
        return binary_logits, error_logits

# Loss
def mtl_loss(binary_logits, error_logits, binary_labels, error_labels, weight_error=0.5):
    # BCEWithLogitsLoss
    loss_binary = nn.BCEWithLogitsLoss()(binary_logits, binary_labels)
    
    # MutliClass loss
    mask = (binary_labels == 0).float()  # mask for error texts
    loss_error = nn.CrossEntropyLoss()(
        error_logits[mask.bool()], 
        error_labels[mask.bool()]
    )
    
    # Sum losses
    total_loss = loss_binary + weight_error * loss_error
    return total_loss, loss_binary, loss_error

# Train
def train(model, dataloader, optimizer, scheduler, device, writer, epoch):
    model.train()
    total_loss = 0
    
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        binary_labels = batch['binary_label'].to(device)
        error_labels = batch['error_label'].to(device)
        
        binary_logits, error_logits = model(input_ids, attention_mask)
        loss, loss_bin, loss_err = mtl_loss(
            binary_logits, error_logits, 
            binary_labels, error_labels
        )
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Log in tensorboard
        if writer:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar('Loss/train_binary', loss_bin.item(), epoch * len(dataloader) + step)
            writer.add_scalar('Loss/train_error', loss_err.item(), epoch * len(dataloader) + step)
            if scheduler:
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch * len(dataloader) + step)
    
    return total_loss / len(dataloader)

# Validation
def evaluate(model, dataloader, device, writer, epoch, binary_thresh=0.5):
    model.eval()
    total_loss = 0
    binary_preds, binary_labels = [], []
    error_preds, error_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            binary_labels_batch = batch['binary_label'].to(device)
            error_labels_batch = batch['error_label'].to(device)
            
            binary_logits, error_logits = model(input_ids, attention_mask)
            loss, loss_bin, loss_err = mtl_loss(
                binary_logits, error_logits, 
                binary_labels_batch, error_labels_batch
            )
            
            total_loss += loss.item()
            
            # Save prds
            binary_preds.extend(torch.sigmoid(binary_logits).cpu().numpy())
            binary_labels.extend(binary_labels_batch.cpu().numpy())
            error_preds.extend(torch.argmax(error_logits, dim=1).cpu().numpy())
            error_labels.extend(error_labels_batch.cpu().numpy())
    
    # Metrics for binary clf
    binary_preds = [1 if p > binary_thresh else 0 for p in binary_preds]
    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds)
    binary_precision = precision_score(binary_labels, binary_preds)
    binary_recall = recall_score(binary_labels, binary_preds)
    
    # Metrics for multi-class clf
    error_mask = [label != -1 for label in error_labels]  # Exclude correct texts
    error_preds_filtered = [p for p, m in zip(error_preds, error_mask) if m]
    error_labels_filtered = [l for l, m in zip(error_labels, error_mask) if m]
    
    if error_labels_filtered:
        error_accuracy = accuracy_score(error_labels_filtered, error_preds_filtered)
        error_f1 = f1_score(error_labels_filtered, error_preds_filtered, average='macro')
        error_report = classification_report(error_labels_filtered, error_preds_filtered)
    else:
        error_accuracy, error_f1, error_report = 0, 0, "No errors in validation set"
    
    # Log in tensorboard
    if writer:
        writer.add_scalar('Loss/val', total_loss / len(dataloader), epoch)
        writer.add_scalar('Accuracy/binary_val', binary_accuracy, epoch)
        writer.add_scalar('F1/binary_val', binary_f1, epoch)
        writer.add_scalar('Accuracy/error_val', error_accuracy, epoch)
        writer.add_scalar('F1/error_val', error_f1, epoch)
    
    # Metrics output
    print(f"Validation Loss: {total_loss / len(dataloader):.3f}")
    print(f"Binary Accuracy: {binary_accuracy:.3f}, F1: {binary_f1:.3f}")
    print(f"Error Accuracy: {error_accuracy:.3f}, F1: {error_f1:.3f}")
    print("Error Classification Report:")
    print(error_report)
    
    return total_loss / len(dataloader)

# Parse args
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task Learning with Embedding Model")
    parser.add_argument('--model_name', type=str, default='cointegrated/rubert-tiny2', help='Pretrained BERT model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--num_error_types', type=int, default=4, help='Number of error types')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--warmup_ration', type=int, default=0.1, help='Number of warmup steps for scheduler')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--output_model_path', type=str, default='bert_mtl_model.pth', help='Path to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Init tensorboard
    log_dir = os.path.join(args.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Train test-example
    texts = ["Пример 1", "Пример 2"]
    binary_labels = [1, 0]  # 1 -- correct, 0 -- incorrect
    error_labels = [-1, 3]   # 0-3 error type for incorrect, -1 for correct
    
    dataset = TextDataset(texts, binary_labels, error_labels, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
   # Val test-example
    val_texts = ["Пример тест текста 1", "Пример тест текста 2"]
    val_binary_labels = [0, 1]
    val_error_labels = [2, -1]
    
    val_dataset = TextDataset(val_texts, val_binary_labels, val_error_labels, tokenizer, args.max_length)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = ModelMTL(args.model_name, args.num_error_types).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Scheduler
    scheduler = None
    if args.use_scheduler:
        total_steps = len(dataloader) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_steps * total_steps),
            num_training_steps=total_steps
        )
    
    # Train-val cycle
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = train(model, dataloader, optimizer, scheduler, device, writer, epoch)
        print(f"Train Loss: {train_loss:.3f}")
        
        # Val
        val_loss = evaluate(model, val_dataloader, device, writer, epoch)
        print(f"Validation Loss: {val_loss:.3f}")
    
    # Save model
    torch.save(model.state_dict(), args.output_model_path)
    print(f"Model saved to {args.output_model_path}")
    
    writer.close()

if __name__ == "__main__":
    main()