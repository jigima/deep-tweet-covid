import os
from pathlib import Path
from transformers import AutoTokenizer, EarlyStoppingCallback
#from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
#from optimum.onnxruntime.configuration import AutoQuantizationConfig  # has avx2/avx512 presets
#import onnx
#import onnx.numpy_helper
#import onnxruntime as ort
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_from_disk, DatasetDict

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
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)   # pick avx2 for most CPUs; use avx512_vnni if available
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
"""
if __name__ == "__main__":
    # Load the test dataset
    test_dataset = load_from_disk("data/test_dataset")
    # Evaluate the quantized ONNX model
    evaluate_onnx_model(Q8_DIR/"model_quantized.onnx",model_name, test_dataset,
                        text_column=TEXT_COLUMN, label_column=LABEL_COLUMN, batch_size=32)
"""
#----------------------------------------
# Distillation script for training a student model using a teacher model
# This script uses the Hugging Face Transformers library for model training and evaluation.
# It implements a custom Trainer class for knowledge distillation, where the student model learns from both
# the teacher model's predictions and the ground truth labels.
#-----------------------------------------
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np


# Custom data collator for distillation
def custom_data_collator(batch):
    #DEBUG print("Collating batch with keys:", batch[0].keys())
    data = {}
    # Always stack these if present
    for key in ["input_ids", "attention_mask", "teacher_input_ids", "teacher_attention_mask"]:
        if key in batch[0]:
            data[key] = torch.stack([torch.tensor(x[key]) for x in batch])
    # Handle labels
    if "labels" in batch[0]:
        data["labels"] = torch.tensor([x["labels"] for x in batch])
    #DEBUG print(data.keys())
    return data



# Custom Trainer for distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()  # Set teacher model to evaluation mode
        if torch.cuda.is_available():
            self.teacher.to("cuda")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = model.device
        # Move all input tensors to the model's device
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(device)
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        teacher_inputs = {
            "input_ids": inputs["teacher_input_ids"],
            "attention_mask": inputs["teacher_attention_mask"]
        }
        with torch.no_grad():
            outputs_teacher = self.teacher(**teacher_inputs)
        loss_ce = outputs_student.loss
        loss_kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(outputs_student.logits / self.temperature, dim=-1),
            torch.nn.functional.softmax(outputs_teacher.logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        return (loss, outputs_student) if return_outputs else loss


# Prepare TrainingArguments and datasets as usual
args = TrainingArguments(
    output_dir=f"trainer/microsoft-deberta-v3-small taught by-{model_name.replace('/', '-')}-final train",
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,  # saves best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    per_device_eval_batch_size=8,
    report_to="wandb",
    per_device_train_batch_size=8,
    remove_unused_columns=False, #keeps all columns in the dataset
    # Acceleration settings
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch sizes
    gradient_checkpointing=True,  # Trade compute for memory savings
    dataloader_num_workers=4,  # Parallel data loading
    optim="adamw_torch_fused"
)

from train_utils import compute_metrics
if __name__== "__main__":
    train_dataset = load_from_disk("data/train_dataset")
    test_dataset = load_from_disk("data/test_dataset")
    split = train_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="SentimentLabel")
    # Combine into a DatasetDict
    dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
        "test": test_dataset
    })
    # Tokenize datasets
    teacher_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=FAST_TOKEN)
    tc_tokenized = tokenize_and_cache_dataset(dataset, model_name, teacher_tokenizer)
    tc_tokenized = tc_tokenized.rename_columns({
        "input_ids": "teacher_input_ids",
        "attention_mask": "teacher_attention_mask"
    })

    student_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False)
    student_tokenizer.max_model_length = 200  # Set the maximum length for tokenization
    st_tokenized = tokenize_and_cache_dataset(dataset, "microsoft/deberta-v3-small", student_tokenizer)
    # Add teacher inputs to student tokenized datasets
    train_with_teacher = st_tokenized["train"].add_column(
        "teacher_input_ids", tc_tokenized["train"]["teacher_input_ids"]
    ).add_column(
        "teacher_attention_mask", tc_tokenized["train"]["teacher_attention_mask"]
    )
    eval_with_teacher = st_tokenized["eval"].add_column(
        "teacher_input_ids", tc_tokenized["eval"]["teacher_input_ids"]
    ).add_column(
        "teacher_attention_mask", tc_tokenized["eval"]["teacher_attention_mask"]
    )
    test_with_teacher = st_tokenized["test"].add_column(
        "teacher_input_ids", tc_tokenized["test"]["teacher_input_ids"]
    ).add_column(
        "teacher_attention_mask", tc_tokenized["test"]["teacher_attention_mask"]
    )

    # Load teacher and student models
    print("loading models...")
    teacher = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=5)
    student = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=5)

    os.environ["WANDB_PROJECT"] = f"deberta-v3-small-distillation taught by-{model_name.replace('/', '-')}-final train"
    # Use your tokenized datasets
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=train_with_teacher, #distillation_dataset
        eval_dataset=eval_with_teacher,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
        data_collator=custom_data_collator
    )

    trainer.train()
    #evaluate the student model on the test set
    test_metrics = trainer.evaluate(test_with_teacher)
    print("Test set metrics:", test_metrics)
