import os
from pathlib import Path
from transformers import AutoTokenizer, EarlyStoppingCallback

from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score
from datasets import load_from_disk, DatasetDict

from train_utils import tokenize_and_cache_dataset

MODEL_DIR = Path("trainer/deberta-final-train")         # A fine-tuned HF/Transformers model SHOULD BE THE SAME BASE MODEL FOR ALL COMPRESSION SCRIPTS
ONNX_DIR = Path("artifacts/deberta-seqcls-onnx")        # <- must be a different directory
Q8_DIR = Path("artifacts/deberta-seqcls-onnx-int8")
PRUNED_DIR = Path("artifacts/deberta-pruned")
model_name="microsoft/deberta-v3-base" #cardiffnlp/twitter-roberta-base-sentiment  #microsoft/deberta-v3-base
FAST_TOKEN=False                       # for RoBERTa use_fast=True, for DeBERTa use_fast=False (experienced issues with fast tokenizer)
DATASET_PATH = Path("data/test_dataset")
TEXT_COLUMN = "OriginalTweet"         # column with input text
LABEL_COLUMN = "SentimentLabel"       # column with int labels (0..num_labels-1)
QUANTIZATION= False         #enable/disable quantization script
QUANTIZATION_EVAL= False    #enable/disable quantization evaluation script
DISTILLATION= False        #enable/disable distillation script
PRUNING = False            #enable/disable pruning script
PRUNE_AMOUNT = 0.3
PRUNE_EVAL = False
COUNTING = True
COUNTING_DIR= Q8_DIR

#----------------------------------------
# Quantization script for exporting a model to ONNX and quantizing it to INT8
# This script uses the Optimum library for ONNX Runtime and quantization.
# It exports a Hugging Face Transformers model to ONNX format and then quantizes it to INT8.
# The quantized model can be used for inference with reduced memory footprint
#-----------------------------------------

if QUANTIZATION:
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig  # has avx2/avx512 presets
    import onnx
    import onnx.numpy_helper

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
        onnx_sum = onnx_weight_sum(ONNX_DIR/"model.onnx")
        q8_sum = onnx_weight_sum(Q8_DIR/"model_quantized.onnx")
        print(f"Exported ONNX sum: {onnx_sum:.2f}")
        print(f"Quantized INT8 sum: {q8_sum:.2f}")

    if __name__ == "__main__":
        quantization()
if QUANTIZATION_EVAL:
    import onnxruntime as ort
    import numpy as np

    def evaluate_onnx_model(onnx_model_path, model_name, test_dataset, text_column=TEXT_COLUMN,
                            label_column=LABEL_COLUMN, batch_size=32):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=FAST_TOKEN)
        session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        input_names = {inp.name for inp in session.get_inputs()}

        all_preds = []
        all_labels = []
        all_probs = []

        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]
            texts = batch[text_column]
            labels = batch[label_column]
            inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=200)
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in input_names}
            outputs = session.run(None, ort_inputs)
            logits = outputs[0]
            preds = np.argmax(logits, axis=1)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax

            all_preds.extend(preds.tolist())
            all_labels.extend(labels)
            all_probs.extend(probs.tolist())

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        print(f"Quantized model accuracy: {accuracy:.4f}")
        print(f"Quantized model macro F1: {f1:.4f}")
        print(f"Quantized model macro Recall: {recall:.4f}")
        print(f"Quantized model macro AUC: {auc:.4f}")

    if __name__ == "__main__":
        # Load the test dataset
        test_dataset = load_from_disk(DATASET_PATH)
        # Evaluate the quantized ONNX model
        evaluate_onnx_model(Q8_DIR/"model_quantized.onnx",model_name, test_dataset,
                            text_column=TEXT_COLUMN, label_column=LABEL_COLUMN, batch_size=32)

#----------------------------------------
# Distillation script for training a student model using a teacher model
# This script uses the Hugging Face Transformers library for model training and evaluation.
# It implements a custom Trainer class for knowledge distillation, where the student model learns from both
# the teacher model's predictions and the ground truth labels.
#-----------------------------------------

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

if DISTILLATION:
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
            train_dataset=train_with_teacher,
            eval_dataset=eval_with_teacher,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
            data_collator=custom_data_collator
        )

        trainer.train()
        #evaluate the student model on the test set
        test_metrics = trainer.evaluate(test_with_teacher)
        print("Test set metrics:", test_metrics)


#-----------------------------
# Pruning script for reducing model size by removing less important weights
#the pruning process involves identifying and removing weights that contribute less to the model's performance.
#-----------------------------
if PRUNING:
    import torch.nn.utils.prune as prune
    import torch
    from transformers import AutoModelForSequenceClassification
    def prune_model(model, amount=0.2):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
        return model


    def main():
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model = prune_model(model, amount=PRUNE_AMOUNT)
        PRUNED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(PRUNED_DIR)
        print(f"Pruned model saved to {PRUNED_DIR}")


    if __name__ == "__main__":
        main()

if PRUNE_EVAL:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
    from datasets import load_from_disk, DatasetDict
    from train_utils import tokenize_and_cache_dataset, compute_metrics

    def main():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(PRUNED_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=FAST_TOKEN)
        test_dataset = load_from_disk(DATASET_PATH)

        # Tokenize test set
        tokenized = tokenize_and_cache_dataset(
             DatasetDict({"test": test_dataset}),
            model_name=model_name,
            tokenizer=tokenizer,
            num_proc=1,
            caching=False
        )["test"]

        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate(tokenized)
        print(metrics)


    if __name__ == "__main__":
        main()
#-----------------------------
# model size and parameter count
#-----------------------------

if COUNTING:
    import torch
    from transformers import AutoModelForSequenceClassification


    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
        return total, nonzero


    def onnx_nonzero_weight_count(path):
        import onnx
        import onnx.numpy_helper

        m = onnx.load(path)
        return sum((onnx.numpy_helper.to_array(init) != 0).sum() for init in m.graph.initializer)

    def get_model_size(model_dir):
        total_size = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
        return total_size / (1024 * 1024)  # MB

    def main():
        if COUNTING_DIR == Q8_DIR or COUNTING_DIR == ONNX_DIR:
            # Quantized ONNX model
            if COUNTING_DIR == Q8_DIR:
                onnx_path = COUNTING_DIR / "model_quantized.onnx"
            else: onnx_path = COUNTING_DIR / "model.onnx"
            onnx_sum = onnx_nonzero_weight_count(onnx_path)
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"ONNX quantized model file size: {size_mb:.2f} MB")
            print(f"ONNX quantized model weight count: {onnx_sum:.2f}")
        else:
            # Standard Hugging Face model
            model = AutoModelForSequenceClassification.from_pretrained(COUNTING_DIR)
            total, nonzero = count_parameters(model)
            size_mb = get_model_size(COUNTING_DIR)
            print(f"Total parameters: {total:,}")
            print(f"Non-zero parameters: {nonzero:,}")
            print(f"Model file size: {size_mb:.2f} MB")


    if __name__ == "__main__":
        main()