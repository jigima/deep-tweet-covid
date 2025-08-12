import numpy as np
from datasets import load_dataset, DatasetDict
import wandb
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
#this is the correct value but the attribute wasn't updated in the HF repository, so we set it manually
tokenizer.model_max_length = 512  # Set the maximum length for tokenization

def tokenize_function(examples):
    return tokenizer(examples["OriginalTweet"], padding="max_length", truncation=True)



from transformers import AutoModelForSequenceClassification

#model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli",num_labels=np.unique(tokenized_datasets["train"]["Sentiment"]).size)

import evaluate
from sklearn.metrics import recall_score

metric_acc = evaluate.load("accuracy")
metric_auc = evaluate.load("roc_auc")
metric_recall = evaluate.load('recall')
metric_f1 = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric_acc.compute(predictions=predictions, references=labels)
    auc = metric_auc.compute(predictions=predictions, references=labels)
    recall = metric_recall.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "auc": auc["roc_auc"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import optuna
def model_init():
    # Return a new model instance for each trial
    return AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli",
                                                              num_labels=5)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [32, 64, 128, 256]),
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
        save_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        # Acceleration settings
        fp16=True,                         # Enable mixed precision training
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