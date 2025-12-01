import os
os.environ['HF_HOME'] = "/mnt/d/RFP_Finetuning/hf_cache/"
from datasets import load_dataset 
from unsloth import FastLanguageModel
from trl import SFTTrainer , SFTConfig
import torch
from custom_prompts import EA_prompt , proposal_prompt , rfp_prompt



max_seq_length = 2048  # Choose any! Support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.



checkpoint = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

model , tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
    # device_map="auto",
    # max_memory={0: "80GB"},
)
print("Getting base model ...")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
print("Loading peft model ...")


EOS_TOKEN = tokenizer.eos_token

def format_prompt_fucnc(examples):


    document_text = examples['text']
    document_type = examples['type']

    texts = []
    for doc_text , doc_type in zip(document_text, document_type):

        if doc_type == "EA":
            text = EA_prompt.format(doc_text)
            texts.append(text)
        elif doc_type =="proposal":
            text = proposal_prompt.format(doc_text)
            texts.append(text)
        else:
            text = rfp_prompt.format(doc_text)
            texts.append(text)
    return texts


dataset = load_dataset("csv" , data_files="Dataset/combined_data.csv")


split_dataset = dataset['train'].train_test_split(test_size=0.2)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    formatting_func=format_prompt_fucnc,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    args=SFTConfig(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        warmup_steps=5,
        num_train_epochs=4,
        max_steps=10000,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear", 
        seed=3407,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="./Result/outputs",
    )
)

trainer_stats = trainer.train()


print("Training done ...")


print("Saving in 16 bit format ...")
# Save model in 8bit precision
model.save_pretrained_merged(
    "16_bit_trained",  # folder where model will be saved
    tokenizer,
    save_method="merged_16bit"
)


print("Saving in 4 bit ...")
model.save_pretrained_merged("model_4_bit", tokenizer, save_method = "merged_4bit_forced",)

print("Done ...")