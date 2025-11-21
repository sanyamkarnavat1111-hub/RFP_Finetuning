import os
os.environ['HF_HOME'] = "/mnt/d/RFP_Finetuning/hf_cache/"
import pandas as pd
from transformers import AutoTokenizer , AutoModelForSequenceClassification , AutoConfig
from datasets import load_dataset , DatasetDict
import evaluate
import torch
from transformers import TrainingArguments, Trainer


accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)


dataset = load_dataset("csv" , data_files="extracted_file.csv")


prompt = """
### EA Requirements:
{}

### RFP Coverage:
{}

### Gap Analysis:
{}
"""

def combine_text(example):
    return {
        "text": f"EA_Requirement: {example['EA_Requirement']} [SEP] "
                f"RFP_Coverage: {example['RFP_Coverage']}"
                f"Gap_analysis: {example['Status']} [SEP] ",
        "label": example["Status"]
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
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)
tokenized_datasets = DatasetDict({
    "train": split_dataset["train"],
    "eval": split_dataset["test"]
})

# Update training arguments to evaluate
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    eval_strategy="steps",  
    eval_steps=50
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"], 
    processing_class=tokenizer, # Our tokenizer
    compute_metrics=compute_metrics
)


trainer.train()


# Save the trained model and tokenizer locally
model_save_path = "./saved_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")
