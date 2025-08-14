import numpy as np
from datasets import load_dataset, DatasetDict
from optuna.pruners import NopPruner
import wandb
from transformers import AutoTokenizer
from train_utils import *



#this is the correct value but the attribute wasn't updated in the HF repository, so we set it manually


# Define the hyperparameter search space
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8,16,32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]),
        #"num_layers_finetune":  trial.suggest_categorical("num_layers", [0 ,4, 6, 8, 12]) #0 means all layers are trainable
    }

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

#multiprocessing shield to avoid spawning multiple processes in the same script
if __name__ == "__main__":
    # Load train and test datasets separately
    train_dataset = load_dataset("csv", data_files="data/Corona_NLP_train_cleaned.csv")["train"]
    #test_dataset = load_dataset("csv", data_files="data/Corona_NLP_test_cleaned.csv")["train"]

    # Split train into train/eval (e.g., 70% train, 30% eval)
    split = train_dataset.train_test_split(test_size=0.3, seed=42)
    # Combine into a DatasetDict
    dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"]
    })

    #model_name="cardiffnlp/twitter-roberta-base-sentiment"
    model_name = "microsoft/deberta-v3-base"
    from transformers import DebertaV2Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    #tokenizer = DebertaV2Tokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.model_max_length = 512  # Set the maximum length for tokenization
    # Tokenize the datasets
    tokenized_datasets=tokenize_and_cache_dataset(dataset, model_name, tokenizer)

    # Get wandb configuration
    wandb_config = get_wandb_config(model_name)

    training_args = TrainingArguments(
        output_dir="trainer",
        num_train_epochs=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_eval_batch_size=8,
        report_to="wandb",
        # Acceleration settings
        fp16=True,
        bf16=False, # Enable mixed precision training
        gradient_accumulation_steps=8,     # Accumulate gradients to simulate larger batch sizes
        gradient_checkpointing=True,       # Trade compute for memory savings
        dataloader_num_workers=4,          # Parallel data loading
        optim="adamw_torch_fused"          # Use fused optimizer implementations
    )

    trainer = Trainer(
        model_init=create_model_init(model_name, num_labels=5),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8),
                   CustomWandbCallback(project_name=wandb_config["project"])]
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        n_trials=15,
        backend="optuna",
        pruner=NopPruner()
    )