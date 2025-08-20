from pathlib import Path
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer


# ----------------------------
# Config: adjust these to your project
# ----------------------------
MODEL_DIR = Path("trainer/deberta-v3-base-sentiment-HP-search-best-run/checkpoint-2706/trainer_state.json")  # reads model from here
ONNX_DIR = Path("trainer/deberta-run-7-finetune")  # ONNX model output directory
QUANTIZED_DIR = Path("trainer/deberta-run-7-finetune-quantized")  # quantized model output directory
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


# compression.py


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def export_to_onnx(model_dir, onnx_dir):
    # Export to ONNX using Optimum
    ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        from_transformers=True,
        export=True,
        save_dir=onnx_dir
    )

def quantize_onnx_model(onnx_dir, quantized_dir):
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)

    quantizer.quantize(save_dir=quantized_dir)

if __name__ == "__main__":

    # Load model (optional, for verification)
    model, tokenizer = load_model(MODEL_DIR)

    # Export to ONNX
    export_to_onnx(MODEL_DIR, ONNX_DIR)

    # Quantize ONNX model
    quantize_onnx_model(ONNX_DIR, QUANTIZED_DIR)