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
STATE_JSON = Path("trainer/run-7/checkpoint-2706/trainer_state.json")  # reads best checkpoint from here
OUTPUT_DIR = Path("trainer/deberta-run-7-finetune")  # new run output directory
DATA_FILE = Path("data/train_dataset")# Hugging Face dataset saved with load_from_disk
TEXT_COLUMN = "OriginalTweet"         # column with input text
LABEL_COLUMN = "SentimentLabel"       # column with int labels (0..num_labels-1)
TEST_SIZE = 0.20                      # new eval split ratio
SEED = 64                             # reproducibility
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

if __name__ == "__main__":
    main()