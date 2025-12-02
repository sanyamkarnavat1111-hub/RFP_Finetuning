import os
os.environ['HF_HOME'] = "/mnt/d/RFP_Finetuning/hf_cache/"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
import evaluate
import torch.nn as nn



# ------------------- METRICS FOR REGRESSION -------------------
mse_metric = evaluate.load("mse")
mae_metric = evaluate.load("mae")
pearson_metric = evaluate.load("pearsonr")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze()
    
    mse = mse_metric.compute(predictions=predictions, references=labels)["mse"]
    mae = mae_metric.compute(predictions=predictions, references=labels)["mae"]
    pearson = pearson_metric.compute(predictions=predictions, references=labels)["pearsonr"]
    
    return {"mse": mse, "mae": mae, "pearson": pearson}

# ------------------- MODEL & CONFIG (WITH SIGMOID) -------------------
checkpoint = "google-bert/bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=1
)


original_head = model.classifier
model.classifier = nn.Sequential(
    original_head,
    nn.Sigmoid()
)


print("Sigmoid added to regression head → predictions will be between 0 - 100 %")

# Set regression mode
config = AutoConfig.from_pretrained(checkpoint)
config.problem_type = "regression"
config.num_labels = 1
model.config = config

# ------------------- DATA PREP -------------------
dataset = load_dataset("csv", data_files="/mnt/d/RFP_Finetuning/Backup_files/Dataset/extracted_file_normalized.csv")

def combine_text(example):
    return {
        "text": f"EA_Requirement: {example['EA_Requirement']} [SEP] RFP_Coverage: {example['RFP_Coverage']}",
        "label": float(example["score_normalized"])   # must be 0.0 – 1.0
    }

dataset = dataset.map(combine_text)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
split = tokenized_datasets["train"].train_test_split(test_size=0.2, seed=42)
tokenized_datasets = DatasetDict({
    "train": split["train"],
    "eval": split["test"]
})

# ------------------- TRAINING ARGS -------------------
training_args = TrainingArguments(
    output_dir="./results_regression",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="pearson",
    greater_is_better=True,
    logging_dir="./logs",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ------------------- TRAIN -------------------
trainer.train()

print("Best model checkpoint:", trainer.state.best_model_checkpoint)
print("Best pearson:", trainer.state.best_metric)

# ------------------- SAVE -------------------
model_save_path = "./saved_model_regression"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model with Sigmoid saved to {model_save_path}")