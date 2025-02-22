import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import argparse

MAX_LENGTH = 128
NUM_ERRORS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_MODEL_PATH = "cointegrated/rubert-tiny2"
WEIGHTS_MODEL_PATH = "./bert_mtl_model.pth"
TOKENIZER_PATH = "cointegrated/rubert-tiny2"
OUTPUT_FILE = "predictions.txt"

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

def predict(text, model, tokenizer, device, binary_thresh=0.5):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        binary_logits, error_logits = model(input_ids=encoding['input_ids'].to(DEVICE),
                                            attention_mask=encoding['attention_mask'].to(DEVICE),)
    
    is_correct = torch.sigmoid(binary_logits).item() > binary_thresh
    error_type = torch.argmax(error_logits).item() if is_correct != 1 else -1

    return is_correct, error_type

def main():
    parser = argparse.ArgumentParser(description="Model for correctness and type error prediction")
    parser.add_argument("--text", type=str, required=True, help="Target text.")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="File for save prediction results.")
    args = parser.parse_args()

    model = ModelMTL(model_name=INIT_MODEL_PATH, num_error_types=NUM_ERRORS)
    model.load_state_dict(torch.load(WEIGHTS_MODEL_PATH, weights_only=True))
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    is_correct, error_type = predict(args.text, model, tokenizer, DEVICE)

    result = f"Text: {args.text}\n"
    result += f"Correct: {'Yes' if is_correct else 'No'}\n"
    if not is_correct:
        result += f"Error type: {error_type}\n"

    print(result)

    with open(args.output, "a") as f:
        f.write(result + "\n")

if __name__ == "__main__":
    main()