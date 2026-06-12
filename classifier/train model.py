"""
sLoRA fine-tuning of sentence-transformers/all-MiniLM-L6-v2 for text classification.

Inputs:
  inputs.json         – [{"id": ..., "title": "..."}, ...]
  classifications.json – [{"id": ..., "classification": "..."}, ...]

Output:
  slora_model/        – saved LoRA adapter + tokenizer
  slora_model_merged/ – full merged model (adapter baked in)

Install dependencies first:
  pip install torch transformers peft scikit-learn accelerate
"""

import json
import os
from collections import defaultdict

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
INPUTS_FILE     = "inputs.json"
LABELS_FILE     = "classifications.json"
ADAPTER_DIR     = "slora_model"
MERGED_DIR      = "slora_model_merged"

# Classification labels — replace with your real taxonomy as needed
LABEL_LIST = [
    "Economics & Finance",
    "Health & Medicine",
    "Environment & Climate",
    "Education & Training",
    "Demographics & Population",
    "Housing & Real Estate",
    "Labour & Employment",
    "Trade & Industry",
    "Crime & Justice",
    "Science & Technology",
]

# Training hyper-params
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-4
TEST_SPLIT   = 0.15
RANDOM_SEED  = 42

# LoRA hyper-params
LORA_R       = 8    # rank
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1
# Target the query & value projections inside each attention layer
LORA_TARGETS = ["query", "value"]

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(inputs_path: str, labels_path: str):
    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Index labels by id
    label_by_id = {str(item["id"]): item["classification"] for item in labels}

    texts, raw_labels = [], []
    skipped = 0
    for item in inputs:
        item_id = str(item["id"])
        title   = item.get("title", "").strip()
        label   = label_by_id.get(item_id)
        if not title or label is None:
            skipped += 1
            continue
        # Normalize label to the closest known label (case-insensitive exact match first)
        matched = next((l for l in LABEL_LIST if l.lower() == label.lower()), None)
        if matched is None:
            # Fall back: pick first label containing a keyword from the raw label
            words = label.lower().split()
            matched = next(
                (l for l in LABEL_LIST if any(w in l.lower() for w in words)),
                LABEL_LIST[0],  # default to first label if nothing matches
            )
        texts.append(title)
        raw_labels.append(matched)

    if skipped:
        print(f"  Skipped {skipped} records (missing title or label).")

    return texts, raw_labels


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class TitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        self.encodings = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor([label2id[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ──────────────────────────────────────────────
# MODEL  (LoRA encoder + classification head)
# ──────────────────────────────────────────────

class LoraClassifier(nn.Module):
    def __init__(self, base_model, num_labels: int, hidden_size: int = 384):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
        )

    def mean_pool(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, id2label):
    model.eval()
    all_preds, all_true = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        _, logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    preds_named = [id2label[p] for p in all_preds]
    true_named  = [id2label[t] for t in all_true]
    print(classification_report(true_named, preds_named, zero_division=0))
    correct = sum(p == t for p, t in zip(all_preds, all_true))
    return correct / len(all_true)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load & split data ──
    print("Loading data...")
    texts, raw_labels = load_data(INPUTS_FILE, LABELS_FILE)
    print(f"  {len(texts)} samples loaded across {len(set(raw_labels))} classes.\n")

    label2id = {l: i for i, l in enumerate(LABEL_LIST)}
    id2label = {i: l for l, i in label2id.items()}

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, raw_labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=raw_labels
    )
    print(f"  Train: {len(train_texts)}  |  Val: {len(val_texts)}\n")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = TitleDataset(train_texts, train_labels, tokenizer, label2id)
    val_ds   = TitleDataset(val_texts,   val_labels,   tokenizer, label2id)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ── Base encoder + LoRA ──
    print("Applying LoRA to base model...")
    base_encoder = AutoModel.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    lora_encoder = get_peft_model(base_encoder, lora_config)
    lora_encoder.print_trainable_parameters()

    model = LoraClassifier(lora_encoder, num_labels=len(LABEL_LIST)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ── Training ──
    print(f"\nTraining for {EPOCHS} epoch(s)...\n")
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_dl, optimizer, device)
        print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}")
        acc = evaluate(model, val_dl, device, id2label)
        print(f"  Val accuracy: {acc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            # Save LoRA adapter weights only
            os.makedirs(ADAPTER_DIR, exist_ok=True)
            model.encoder.save_pretrained(ADAPTER_DIR)
            tokenizer.save_pretrained(ADAPTER_DIR)
            # Save classifier head separately
            torch.save(model.classifier.state_dict(), os.path.join(ADAPTER_DIR, "classifier_head.pt"))
            print(f"  ✓ New best ({best_acc:.4f}) — adapter saved to '{ADAPTER_DIR}'\n")

    # ── Merge & save full model ──
    print("Merging LoRA weights into base model...")
    merged_encoder = model.encoder.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_encoder.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    torch.save(model.classifier.state_dict(), os.path.join(MERGED_DIR, "classifier_head.pt"))

    # Save label map alongside the model
    with open(os.path.join(MERGED_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    print(f"Merged model saved to '{MERGED_DIR}'.")
    print(f"\nDone. Best val accuracy: {best_acc:.4f}")


# ──────────────────────────────────────────────
# INFERENCE HELPER  (import and call separately)
# ──────────────────────────────────────────────

def predict(titles: list[str], model_dir: str = MERGED_DIR, device_str: str = "cpu") -> list[str]:
    """Load the saved model and classify a list of title strings."""
    device = torch.device(device_str)

    with open(os.path.join(model_dir, "label_map.json")) as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    num_labels = len(id2label)

    tokenizer    = AutoTokenizer.from_pretrained(model_dir)
    base_encoder = AutoModel.from_pretrained(model_dir)

    model = LoraClassifier(base_encoder, num_labels=num_labels).to(device)
    model.classifier.load_state_dict(
        torch.load(os.path.join(model_dir, "classifier_head.pt"), map_location=device)
    )
    model.eval()

    enc = tokenizer(titles, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        _, logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return [id2label[i] for i in logits.argmax(dim=-1).tolist()]


if __name__ == "__main__":
    main()