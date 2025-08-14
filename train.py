import numpy as np
from datasets import load_dataset, DatasetDict
from optuna.pruners import MedianPruner, NopPruner
from optuna.testing.objectives import pruned_objective

import wandb
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment",use_fast=True)
#this is the correct value but the attribute wasn't updated in the HF repository, so we set it manually
tokenizer.model_max_length = 256  # Set the maximum length for tokenization

def tokenize_function(examples):
    result = tokenizer(examples["OriginalTweet"], padding="max_length", truncation=True)
    # Add labels to the tokenized output
    result["labels"] = examples["SentimentLabel"]
    return result



from transformers import AutoModelForSequenceClassification



import evaluate
# Use sklearn for AUC calculation
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

metric_acc = evaluate.load("accuracy")
metric_recall = evaluate.load('recall')
metric_f1 = evaluate.load('f1')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Regular metrics
    accuracy = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    recall = metric_recall.compute(predictions=predictions, references=labels, average="macro")

    # Convert logits to probabilities
    probs = softmax(logits, axis=1)

    # Calculate multi-class AUC
    auc = roc_auc_score(
        y_true=labels,
        y_score=probs,
        multi_class='ovr',
        average='macro'
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "recall": recall["recall"],
        "auc": auc
    }

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import optuna
def model_init():
    # Return a new model instance for each trial
    return AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment",
                                                              num_labels=5,
                                                              ignore_mismatched_sizes=True)

def model_init_with_freezing(trial=None):
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment",
        num_labels=5,
        ignore_mismatched_sizes=True
    )

    # Apply layer freezing if in hyperparameter search
    if trial is not None:
        num_layers = trial.params.get("num_layers_finetune", 0)

        # If num_layers_finetune is 0, all layers remain trainable (default)
        if num_layers > 0:
            # First freeze all parameters in the base model
            for param in model.roberta.parameters():
                param.requires_grad = False

            # Make the specified number of encoder layers trainable
            for param in model.roberta.encoder.layer[-num_layers:].parameters():
                param.requires_grad = True

            # Always keep the classifier head trainable
            for param in model.classifier.parameters():
                param.requires_grad = True




    return model
def estimate_optimal_batch_size(
    model_name="cardiffnlp/twitter-roberta-base-sentiment",
    max_length=256,
    num_labels=5,
    max_memory_fraction=0.80,
    gradient_checkpointing=True,
    fp16=True,
    target_batch_size=None
):
    """
    Estimates the maximum safe batch size based on GPU memory constraints.
    """
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("No GPU available, returning default settings")
        return {"batch_size": 8, "vram_usage": "N/A"}

    # Get GPU info
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_memory / (1024**3)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_gb:.2f} GB")
    print(f"Using max {max_memory_fraction*100:.0f}% of VRAM")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # Configure model settings
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Move model to GPU with desired precision
    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(device, dtype=dtype)

    # Create sample input (batch size of 1)
    sample_text = "This is a sample tweet to estimate memory usage."
    encoded = tokenizer([sample_text], padding="max_length",
                      truncation=True, max_length=max_length,
                      return_tensors="pt")

    # Move input to device
    sample_input = {k: v.to(device) for k, v in encoded.items()}
    sample_input["labels"] = torch.tensor([0], device=device)

    # Measure memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    with torch.autocast(device_type="cuda", dtype=dtype) if fp16 else torch.no_grad():
        outputs = model(**sample_input)

    # Backward pass
    loss = outputs.loss
    loss.backward()

    # Get memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    model.zero_grad()

    # Add overhead for optimizer states and buffers (approximately 3x)
    memory_per_sample = peak_memory * 3

    # Maximum memory we want to use
    max_memory = total_memory * max_memory_fraction

    # Calculate max batch size
    max_batch_size = max(1, int(max_memory / memory_per_sample))

    # Round to power of 2 for better GPU efficiency
    power_of_2 = 2 ** int(np.log2(max_batch_size))
    recommended_batch_size = power_of_2

    # Calculate gradient accumulation if target_batch_size is specified
    grad_accumulation_steps = 1
    if target_batch_size and target_batch_size > recommended_batch_size:
        grad_accumulation_steps = max(1, target_batch_size // recommended_batch_size)

    # Print batch size options table
    print("\n--- Batch Size vs. Estimated VRAM Usage ---")
    print(f"{'Batch Size':<10} | {'VRAM (GB)':<10} | {'% of GPU':<10}")
    print("-" * 35)

    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        if bs > max_batch_size * 2:
            break
        mem_usage = (memory_per_sample * bs) / (1024**3)
        percent = (mem_usage / total_gb) * 100
        status = "✓" if bs <= recommended_batch_size else "✗"
        print(f"{bs:<10} | {mem_usage:<10.2f} | {percent:<8.1f}% {status}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        "batch_size": recommended_batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "effective_batch_size": recommended_batch_size * grad_accumulation_steps,
        "vram_usage_gb": (memory_per_sample * recommended_batch_size) / (1024**3),
        "total_vram_gb": total_gb
    }

# Example usage:
# settings = estimate_optimal_batch_size(target_batch_size=128)
# training_args.per_device_train_batch_size = settings["batch_size"]
# training_args.gradient_accumulation_steps = settings["gradient_accumulation_steps"]

# Define the hyperparameter search space
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8,16,32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]),
        "num_layers_finetune":  trial.suggest_categorical("num_layers", [0 ,4, 6, 8, 12]) #0 means all layers are trainable
    }


#multi-processing shield to avoid spawning multiple processes in the same script
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
    # Tokenize the datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True,
                                     num_proc=4,
                                     cache_file_names={
                                         "train": "cache/train_tokenized.arrow",
                                         "eval": "cache/eval_tokenized.arrow"
                                     })

    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="corona-tweet-sentiment", name="optuna-search")

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
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        n_trials=15,
        backend="optuna",
        pruner=NopPruner()
    )