import numpy as np
from datasets import load_dataset, DatasetDict
from optuna.pruners import MedianPruner, NopPruner
from optuna.testing.objectives import pruned_objective

import wandb
from transformers import AutoTokenizer, PreTrainedModel


# Data tokenization function
def tokenize_function(tokenizer,examples):
    result = tokenizer(examples["OriginalTweet"], padding="max_length", truncation=True,max_length=200)
    # Add labels to the tokenized output
    result["labels"] = examples["SentimentLabel"]
    return result

def tokenize_and_cache_dataset(dataset, model_name, tokenizer, num_proc=4,caching=True):
    """
    Tokenizes dataset and caches results with model-specific filenames.

    Args:
        dataset: The dataset to tokenize
        model_name: Name of the model (used for cache file naming)
        tokenizer: The tokenizer to use
        num_proc: Number of processes for parallel tokenization
        caching: Whether to cache the tokenized dataset
    """
    # Create a model-specific identifier for cache files
    model_id = model_name.split('/')[-1]  # Extract last part of model path

    cache_files = {
        "train": f"cache/{model_id}_train_tokenized.arrow",
        "eval": f"cache/{model_id}_eval_tokenized.arrow",
        "test": f"cache/{model_id}_test_tokenized.arrow"
    }

    # Use a function that already has the tokenizer
    def _tokenize_function(examples):
        return tokenize_function(tokenizer, examples)
    if caching:
        return dataset.map(_tokenize_function, batched=True,
                          num_proc=num_proc,
                          cache_file_names=cache_files)
    else:
        return dataset.map(_tokenize_function, batched=True,num_proc=num_proc)

def analyze_token_lengths(dataset, tokenizer, column_name="text"):
    """Analyze token length distribution in dataset"""
    lengths = []

    # Process each example
    for example in dataset["train"]:
        tokens = tokenizer(example[column_name], truncation=False, padding=False)
        lengths.append(len(tokens["input_ids"]))

    # Calculate statistics
    lengths_array = np.array(lengths)
    print(f"Mean length: {lengths_array.mean():.2f}")
    print(f"Median length: {np.median(lengths_array):.2f}")
    print(f"95th percentile: {np.percentile(lengths_array, 95):.2f}")
    print(f"99th percentile: {np.percentile(lengths_array, 99):.2f}")
    print(f"Max length: {lengths_array.max()}")

    # Calculate truncation percentages
    for length in [128, 256, 384, 512]:
        truncated = sum(l > length for l in lengths)
        print(f"Length {length}: {truncated/len(lengths)*100:.2f}% would be truncated")

    return lengths_array

#metrics computation
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

#model intialization functions
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch import nn

#eperimental enhanced model classifier intialization function
def model_init_enhanced(model_name, num_labels=5) -> PreTrainedModel:
    """
    Initializes a sequence classification model with an enhanced classifier head.

    Args:
        model_name (str): The name of the pre-trained model.
        num_labels (int): Number of labels for classification.

    Returns:
        PreTrainedModel: A model instance with a custom classifier head.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    hidden = model.config.hidden_size
    model.classifier = nn.Sequential(
        nn.Linear(hidden, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, model.config.num_labels),
    )

    return model
def create_model_init_enhanced(model_name, num_labels=5):
    """
    Creates a model initialization function for hyperparameter search.

    Args:
        model_name (str): The name of the pre-trained model.
        num_labels (int): Number of labels for classification.

    Returns:
        function: A function that initializes the model with an enhanced classifier head.
    """
    def _model_init():
        return model_init_enhanced(model_name, num_labels)

    return _model_init

def model_init(model_name,num_labels=5)->PreTrainedModel:
    # Return a new model instance for each trial
    return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                              num_labels=num_labels,
                                                              ignore_mismatched_sizes=True)

def model_init_with_freezing(model_name,num_labels=5,trial=None):
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
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

        frozen, trainable = count_frozen_layers(model)
        print(f"[DEBUG] Freezing applied: {frozen} frozen, {trainable} trainable encoder layers")

        # Save into trial attributes for later (DOES NOT WORK YET)
        trial.set_user_attr("num_layers_finetune", num_layers)
        trial.set_user_attr("layers_frozen", frozen)
        trial.set_user_attr("layers_trainable", trainable)

    return model

def create_model_init(model_name, num_labels=5):
    """Creates a model_init function that Trainer can use for hyperparameter search."""
    def _model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    return _model_init

def create_model_init_with_freezing(model_name, num_labels=5, trial=None):
    """Creates a model_init function that allows layer freezing during hyperparameter search."""
    def _model_init():
        return model_init_with_freezing(model_name, num_labels, trial)
    return _model_init

#create a custom Trainer class that passes trial to model_init
from transformers import Trainer

class OptunaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial = None

    def set_trial(self, trial):
        # This will be called by hyperparameter search
        self._trial = trial
        # Attach trial params to training args for callbacks
        if hasattr(self.args, "trial_params") is False:
            self.args.trial_params = {}
        self.args.trial_params.update(trial.params)



# define your model_init function to accept the trial
def model_init_for_optuna(trial, model_name, num_labels=5):
    return model_init_with_freezing(model_name, num_labels=num_labels, trial=trial)
#usade in OptunaTrainer, model_init=lambda trial: model_init_for_optuna(trial, model_name="model_name")

# experimental function to count frozen vs trainable layers for debugging purposes
def count_frozen_layers(model):
    """Count how many encoder layers are frozen vs trainable."""
    frozen, trainable = 0, 0
    for i, layer in enumerate(model.roberta.encoder.layer):
        # Check one parameter from each layer to infer its state
        requires_grad = next(layer.parameters()).requires_grad
        if requires_grad:
            trainable += 1
        else:
            frozen += 1
    return frozen, trainable

# experimental function to estimate optimal maximum batch size based on GPU memory
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

# Weights and Biases integration
def get_wandb_config(model_name):
    """Generate wandb configuration based on model name"""
    # Extract model id for project name
    model_id = model_name.split('/')[-1]

    # Create a timestamp for unique run identification
    from datetime import datetime
    timestamp = datetime.now().strftime("%d%m_%H%M")

    return {"project": f"sentiment-{model_id}-time_{timestamp}"}







