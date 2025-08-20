import numpy as np
from datasets import load_dataset, load_from_disk, DatasetDict
from optuna.pruners import NopPruner
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from train_utils import *
import os
import tempfile
import json
from pathlib import Path
tempfile.tempdir = "D:/temp"
os.environ["HF_HOME"] = "D:/huggingface_cache"
#os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface_cache/datasets"
os.environ["TMPDIR"] = "D:/temp"  # Create this folder if it doesn't exist
os.environ["TEMP"] = "D:/temp"
os.environ["TMP"] = "D:/temp"


# ----------------------------
# Config: adjust these to your project
# ----------------------------
STATE_JSON = Path("trainer/run-7/checkpoint-2706/trainer_state.json")  # reads best checkpoint from here
OUTPUT_DIR = Path("trainer/deberta-final-train")  # new run output directory
DATA_FILE = Path("data/train_dataset")# Hugging Face dataset saved with load_from_disk
model_name="cardiffnlp/twitter-roberta-base-sentiment"
#model_name = "microsoft/deberta-v3-base"
# ----------------------------

def read_best_hp_params(state_json_path: Path) -> str:
    with open(state_json_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    best_hp_params = state["trial_params"]
    # Normalize Windows-style path from JSON for current OS
    return best_hp_params

#multiprocessing shield to avoid spawning multiple processes in the same script
if __name__ == "__main__":
    # Load train and test datasets separately
    train_dataset = load_from_disk("data/train_dataset")
    # Split train into train/eval (e.g., 90% train, 10% eval)
    split = train_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="SentimentLabel")
    #test_dataset = load_dataset("csv", data_files="data/Corona_NLP_test_cleaned.csv")["train"]
    test_dataset = load_from_disk("data/test_dataset")
    # Combine into a DatasetDict
    dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
        "test": test_dataset
    })
    
    # for RoBERTa use_fast=True,
    # for DeBERTa use_fast=False (experienced issues with fast tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = 200  # Set the maximum length for tokenization
    # Analyze token lengths in the dataset
    """token_lengths = analyze_token_lengths(dataset, tokenizer, column_name="OriginalTweet")

    # If you want to visualize (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.hist(token_lengths, bins=50)
    plt.axvline(x=256, color='r', linestyle='--')
    plt.savefig('token_lengths.png')
    plt.show()"""

    # Tokenize the datasets
    tokenized_datasets=tokenize_and_cache_dataset(dataset, model_name, tokenizer)

    # intialize model
    model=model_init(model_name,num_labels=5)
    # Get wandb configuration
    wandb_config = get_wandb_config(model_name)
    os.environ["WANDB_PROJECT"] = wandb_config["project"]

    # Read best hyperparameters from the Optuna state JSON file
    best_hp_params = read_best_hp_params(STATE_JSON)

    training_args = TrainingArguments(
        output_dir=f"trainer/{model_name.replace('/', '-')}-final train",
        num_train_epochs=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2, #saves the last and best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_eval_batch_size=8,
        report_to="wandb",
        per_device_train_batch_size=best_hp_params["per_device_train_batch_size"],
        learning_rate=best_hp_params["learning_rate"],
        weight_decay=best_hp_params["weight_decay"],
        lr_scheduler_type=best_hp_params["lr_scheduler_type"],
        # Acceleration settings
        fp16=True,
        bf16=False, # Enable mixed precision training
        gradient_accumulation_steps=8,     # Accumulate gradients to simulate larger batch sizes
        gradient_checkpointing=True,       # Trade compute for memory savings
        dataloader_num_workers=4,          # Parallel data loading
        optim="adamw_torch_fused"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]

    )

    print("Starting fine-tuning with hyperparameters and early stopping...")
    trainer.train()

    print("Evaluating best checkpoint from this fine-tune run...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    test_metrics = trainer.evaluate(tokenized_datasets["test"])
    print("Test set metrics:", test_metrics)
    # Save the final model and tokenizer
    print(f"Saving final best model to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))