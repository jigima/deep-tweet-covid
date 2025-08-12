import numpy as np
from datasets import load_dataset, DatasetDict
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

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16,32,64]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"])
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
        report_to="wandb",
        # Acceleration settings
        fp16=True,
        bf16=False, # Enable mixed precision training
        gradient_accumulation_steps=4,     # Accumulate gradients to simulate larger batch sizes
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
        backend="optuna"
    )