# python
import os
import json
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from train_utils import *

# ----------------------------
# Config: adjust these to your project
# ----------------------------
STATE_JSON = Path("trainer/deberta-v3-base-sentiment-HP-search-best-run/checkpoint-2706/trainer_state.json")  # reads best checkpoint from here
OUTPUT_DIR = Path("trainer/deberta-run-7-finetune")  # new run output directory
DATA_FILE = Path("data/train_dataset")# Hugging Face dataset saved with load_from_disk
TEXT_COLUMN = "OriginalTweet"         # column with input text
LABEL_COLUMN = "SentimentLabel"       # column with int labels (0..num_labels-1)
TEST_SIZE = 0.10                      # new eval split ratio
SEED = 42                             # reproducibility
LOW_LR = 1.0e-5                       # lower learning rate for careful fine-tuning
WEIGHT_DECAY = 0.16090032972441884    # from best run
BATCH_SIZE = 8                        # from best run
MAX_LENGTH = 200                     # adjust as needed
PATIENCE = 8                          # early stopping patience


def read_best_checkpoint(state_json_path: Path) -> str:
    with open(state_json_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    best_dir = state["best_model_checkpoint"]
    # Normalize Windows-style path from JSON for current OS
    return os.path.normpath(best_dir)

def load_model_and_tokenizer(checkpoint_dir: str):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, config=config)
    return model, tokenizer, config


def main():
    set_seed(SEED)

    checkpoint_dir = read_best_checkpoint(STATE_JSON)
    print(f"Loading best checkpoint from: {checkpoint_dir}")

    model, tokenizer, config = load_model_and_tokenizer(checkpoint_dir)
    tokenizer.model_max_length = MAX_LENGTH  # used by train_utils.tokenize_function

    # Load raw dataset and create a fresh train/eval split
    raw = load_from_disk(str(DATA_FILE))  # a single split dataset
    split = raw.train_test_split(test_size=TEST_SIZE, seed=SEED, stratify_by_column=LABEL_COLUMN)
    ds = DatasetDict(train=split["train"], eval=split["test"])  # match cache keys expected by util
    split2=ds['eval'].train_test_split(test_size=0.5, seed=SEED, stratify_by_column=LABEL_COLUMN)
    ds= DatasetDict(train=split['train'],eval=split2['train'],test=split2['test'])  # match cache keys expected by utils

    # Tokenize and cache using the shared utility (ready for Trainer)
    tokenized = tokenize_and_cache_dataset(
        dataset=ds,
        model_name=config._name_or_path,
        tokenizer=tokenizer,
        num_proc=4,
    )
    # Expose tokenized splits
    train_tok = tokenized["train"]
    eval_tok = tokenized["eval"]
    label_eval_tok = tokenized["test"]  # held-out for labeling procedure
    print(f"Sizes -> train: {len(train_tok)}, finetune_eval: {len(eval_tok)}, label_eval: {len(label_eval_tok)}")

    os.environ["WANDB_PROJECT"] = "deberta-finetune-labeling-procedure-trial-t1808"

    # Training args with lower LR and early stopping
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        greater_is_better=True,

        learning_rate=LOW_LR,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=20,  # will be cut short by early stopping
        lr_scheduler_type="cosine_with_restarts",  # keep same family; change if needed
        warmup_ratio=0.06,

        gradient_accumulation_steps=4,
        fp16=True,
        report_to="wandb",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE, early_stopping_threshold=0)],
    )

    print("Starting fine-tuning with lower learning rate and early stopping...")
    trainer.train()

    print("Evaluating best checkpoint from this fine-tune run...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"Saving final best model to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

#if __name__ == "__main__":
#    main()

#testing the procedure
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from collections import Counter
import evaluate

def compute_procedure_metrics(predictions, labels):
    """
    Computes accuracy and F1 score for already-computed predictions.
    """
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    accuracy = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

def find_optimal_thresholds_by_accuracy(logits: np.ndarray, labels: np.ndarray):
    """
    Finds the optimal probability threshold for each class to maximize one-vs-rest accuracy.

    Args:
        logits: The raw model outputs (logits) from a dataset.
        labels: The true labels for that dataset.

    Returns:
        A dictionary mapping each class index to its optimal threshold.
    """
    print("Finding optimal thresholds by accuracy...")
    num_labels = logits.shape[1]
    probabilities = softmax(logits, axis=1)
    optimal_thresholds = {}

    for class_id in range(num_labels):
        # One-vs-rest labels and probabilities
        y_true = (labels == class_id)
        y_prob = probabilities[:, class_id]

        best_accuracy = 0
        best_threshold = 0.5  # Default threshold

        # Iterate through potential thresholds to find the best one
        for threshold in np.linspace(0.01, 0.99, 1000):
            y_pred = (y_prob >= threshold)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold

        optimal_thresholds[class_id] = best_threshold
        print(f"  - Optimal threshold for class {class_id}: {best_threshold:.4f} (Accuracy: {best_accuracy:.4f})")

    return optimal_thresholds

def calculate_class_frequencies(labels: np.ndarray):
    """
    Calculates the frequency of each class in a set of labels.

    Args:
        labels: An array of true labels.

    Returns:
        A dictionary mapping each class index to its frequency.
    """
    print("Calculating class frequencies...")
    counts = Counter(labels)
    total_samples = len(labels)
    frequencies = {class_id: count / total_samples for class_id, count in counts.items()}
    return frequencies

def apply_custom_procedure(logits: np.ndarray, thresholds: dict, frequencies: dict):
    """
    Applies the custom scoring procedure to a batch of logits.

    Args:
        logits: The raw model outputs for the data to be predicted.
        thresholds: The dictionary of optimal thresholds per class.
        frequencies: The dictionary of class frequencies from the training data.

    Returns:
        An array of final predictions.
    """
    print("Applying custom labeling procedure...")
    probabilities = softmax(logits, axis=1)
    num_samples = logits.shape[0]
    final_predictions = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        sample_probs = probabilities[i]

        # Identify candidate classes that exceed their threshold
        candidate_classes = [
            cid for cid, prob in enumerate(sample_probs) if prob >= thresholds.get(cid, 1.0)
        ]

        if not candidate_classes:
            # Fallback to argmax if no class exceeds its threshold
            final_predictions[i] = np.argmax(sample_probs)
            continue

        # Calculate the sum of frequencies for only the candidate classes
        sum_candidate_freqs = sum(frequencies.get(cid, 0) for cid in candidate_classes)
        if sum_candidate_freqs == 0: # Avoid division by zero
            final_predictions[i] = np.argmax(sample_probs)
            continue

        best_score = -1
        best_class = -1

        # Calculate the score for each candidate class
        for cid in candidate_classes:
            score = (sample_probs[cid] / thresholds[cid]) * (frequencies.get(cid, 0) / sum_candidate_freqs)
            if score > best_score:
                best_score = score
                best_class = cid

        final_predictions[i] = best_class

    return final_predictions

def run_procedure_comparison():
    """
    Loads a trained model and compares the custom labeling procedure
    against the standard argmax method on a held-out test set.
    """
    # --- 1. Configuration & Setup ---
    MODEL_CHECKPOINT = "trainer/deberta-final-train"
    print(f"--- Starting Procedure Comparison using model: {MODEL_CHECKPOINT} ---")
    set_seed(SEED)

    # --- 2. Load Model and Tokenizer ---
    model, tokenizer, config = load_model_and_tokenizer(MODEL_CHECKPOINT)
    tokenizer.model_max_length = MAX_LENGTH

    # --- 3. Load and Prepare Data (reusing logic from main) ---
    raw = load_from_disk(str(DATA_FILE))
    split = raw.train_test_split(test_size=TEST_SIZE, seed=SEED, stratify_by_column=LABEL_COLUMN)
    ds = DatasetDict(train=split["train"], eval=split["test"])
    #split2 = ds['eval'].train_test_split(test_size=0.5, seed=SEED, stratify_by_column=LABEL_COLUMN)
    test_set = load_from_disk("data/test_dataset")  # Load the test dataset
    ds = DatasetDict(train=split['train'], eval=split['train'], test=test_set)

    tokenized = tokenize_and_cache_dataset(
        dataset=ds,
        model_name=config._name_or_path,
        tokenizer=tokenizer,
        num_proc=4,
        caching=False
    )
    train_tok = tokenized["train"]
    eval_tok = tokenized["eval"]  # Used for finding thresholds
    test_tok = tokenized["test"]   # Used for final comparison

    print(f"\nData sizes -> Train: {len(train_tok)}, Eval (for thresholds): {len(eval_tok)}, Test (for comparison): {len(test_tok)}")

    # --- 4. Get Model Predictions (Logits) ---
    # We need a dummy Trainer to run predictions
    trainer = Trainer(model=model,args=TrainingArguments(report_to="none"))

    print("\nGetting predictions for the evaluation set (to find thresholds)...")
    eval_output = trainer.predict(eval_tok)
    eval_logits = eval_output.predictions
    eval_labels = eval_output.label_ids

    print("Getting predictions for the test set (to compare procedures)...")
    test_output = trainer.predict(test_tok)
    test_logits = test_output.predictions
    test_labels = test_output.label_ids

    # --- 5. Run Offline Preparation Steps ---
    # Use the full training set to calculate true class frequencies
    train_labels = np.array(ds["train"][LABEL_COLUMN])
    class_frequencies = calculate_class_frequencies(train_labels)
    print(f"  - Calculated Frequencies: {class_frequencies}")

    # Use the evaluation set (a stand-in for a validation set) to find thresholds
    optimal_thresholds = find_optimal_thresholds_by_accuracy(eval_logits, eval_labels)

    # --- 6. Apply and Compare Procedures on the Test Set ---
    print("\n--- Comparing Procedures on Test Set ---")

    # Standard argmax procedure
    argmax_preds = np.argmax(test_logits, axis=-1)

    # Custom procedure
    custom_preds = apply_custom_procedure(test_logits, optimal_thresholds, class_frequencies)

    # --- 7. Evaluate and Print Metrics ---
    print("\nMetrics for Standard Argmax:")
    argmax_metrics = compute_procedure_metrics(argmax_preds, test_labels)
    for k, v in argmax_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nMetrics for Custom Procedure:")
    custom_metrics = compute_procedure_metrics(custom_preds, test_labels)
    for k, v in custom_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    # main() # You can comment this out if you don't need to re-train
    run_procedure_comparison()