from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import json

#Load Dataset
with open("intent_dataset_bert.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2)

#Tokenizer & Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

#Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

#Training Setup
training_args = TrainingArguments(
    output_dir="./bert-intent-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"]
)

#Train the model
trainer.train()

#Save the model and tokenizer
model.save_pretrained("bert-intent-model")
tokenizer.save_pretrained("bert-intent-model")
