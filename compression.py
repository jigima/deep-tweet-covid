from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig  # has avx2/avx512 presets
import onnx
import onnx.numpy_helper
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_from_disk

from train_utils import tokenize_and_cache_dataset

MODEL_DIR = Path("trainer/deberta-final-train")           # A fine-tuned HF/Transformers model
ONNX_DIR = Path("artifacts/deberta-seqcls-onnx")          # <- must be a different directory
Q8_DIR = Path("artifacts/deberta-seqcls-onnx-int8")
model_name="microsoft/deberta-v3-base"
FAST_TOKEN=False  # for RoBERTa use_fast=True, for DeBERTa use_fast=False (experienced issues with fast tokenizer)
TEXT_COLUMN = "OriginalTweet"         # column with input text
LABEL_COLUMN = "SentimentLabel"       # column with int labels (0..num_labels-1)

def export_to_onnx(model_dir: Path, onnx_dir: Path):
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # 1) Export to ONNX (in memory)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        export=True,                   # do the export
        provider="CPUExecutionProvider"  # change to CUDAExecutionProvider if on GPU
    )
    # 2) Persist the exported model to disk (crucial!)
    ort_model.save_pretrained(onnx_dir)


def quantize_dynamic(onnx_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    # Dynamic quantization (no calibration dataset required)
    qconfig = AutoQuantizationConfig.avx2(is_static=False)   # pick avx2 for most CPUs; use avx512_vnni if available
    quantizer.quantize(quantization_config=qconfig, save_dir=out_dir)



def onnx_weight_sum(path):
    m = onnx.load(path)
    return sum(float(onnx.numpy_helper.to_array(init).sum()) for init in m.graph.initializer)


def quantization():
    export_to_onnx(MODEL_DIR, ONNX_DIR)
    quantize_dynamic(ONNX_DIR, Q8_DIR)
    print(f"Done.\nONNX model: {ONNX_DIR}\nINT8 model: {Q8_DIR}")
    onnx_sum = onnx_weight_sum("artifacts/deberta-seqcls-onnx/model.onnx")
    q8_sum = onnx_weight_sum("artifacts/deberta-seqcls-onnx-int8/model_quantized.onnx")
    print(f"Exported ONNX sum: {onnx_sum:.2f}")
    print(f"Quantized INT8 sum: {q8_sum:.2f}")

#if __name__ == "__main__":
 #   quantization()

def evaluate_onnx_model(onnx_model_path, model_name, test_dataset, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=FAST_TOKEN)
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}

    all_preds = []
    all_labels = []

    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i + batch_size]
        texts = batch[text_column]
        labels = batch[label_column]
        inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=200)
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, ort_inputs)
        preds = np.argmax(outputs[0], axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels)

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Quantized model accuracy: {accuracy:.4f}")
    print(f"Quantized model macro F1: {f1:.4f}")

if __name__ == "__main__":
    # Load the test dataset
    test_dataset = load_from_disk("data/test_dataset")
    # Evaluate the quantized ONNX model
    evaluate_onnx_model(Q8_DIR/"model_quantized.onnx",model_name, test_dataset,
                        text_column=TEXT_COLUMN, label_column=LABEL_COLUMN, batch_size=32)