# Deep Tweet COVID

Analysis of COVID-19 related tweets using NLP techniques for sentiment classification.

## Dataset

Using the [COVID-19 NLP Text Classification Dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data) from Kaggle.

The `EDA.py` script performs exploratory data analysis, producing various visualizations of the dataset in question.
Token length distribution has been analyzed seperately and we have found that a length of 200 is enough to cover the entire dataset.

## Project Overview

This repository contains a solution for COVID-19 tweet sentiment analysis using transformer models. The project includes:

1. **HuggingFace Training Pipeline** - Training, compression, and comparison using HuggingFace's native tools
2. **Custom Training Pipeline** - Training, compression, and comparison using custom implementation

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required dependencies
```

## HuggingFace Training Pipeline

The HuggingFace pipeline uses the `Trainer` API for model fine-tuning and evaluation. A configuration and training script is provided for both RoBERTa and DeBERTa models.
We reccommend setting up an config file for the accelerate library, you can do so with the command:
```bash
  accelerate config
  ```
reccomended settings are using fp16, the appropriate backend for torch dynamo for your hardware (eager works fine on Windows)
### Dataset Preparation
Preprocess the dataset using the provided file
```bash
python "data preprocessing.py"
```
the resulting files will be saved in the `data/test_dataset` and `data/train_dataset` directories.
### Training

Best models from hyperparameter search studies were saved in subdirectories under `trainer/{model_name}-HP-search-best-run/`.

**Steps:**
1. Edit `train.py` to set your desired model and parameters, extract the path to the `trainer_state.json` containing the hyperparameters from the directory mentioned above corresponding the model that you want to train.
2. Edit the output path to the desired location to save the trained model, edit `model_name` variable to the model you want to train (either `cardiffnlp/twitter-roberta-base-sentiment` or `microsoft/deberta-v3-base`).
3. Check that you are using the correct fast tokenization option according to the model. Run the script:
   ```bash
   python train.py
   ``` 
    alternatively you can run using accelerate:
    ```bash
    accelerate launch train.py
    ```

### Compression
the compression techniques use multiple tools from both HuggingFace libraries and torch libraries.

**steps:**
1. Edit flags at the top of the `compression.py` script to choose the model and compression techniques you want to apply.
2. Run the script:
   ```bash
   python compression.py
   ```
**make sure to edit the paths to the model you want to compress, MODEL_DIR variable needs to point to the directory containing the best model you want to compress.**

#### Quantization
Quantization is performed using HuggingFace's `optimum` library with ONNX infrastructure.
you could get warnings during some import operations, those are an API issue and will hopefully be fixed in future releases.
`Quantization` flag enables/disables cration of ONNX and quantized ONNX models from the mdoel in `MODEL_DIR`
check flag `QUANTIZATION_EVAL` to enable/disable quantized model evaluation on the cpu without retraining. note that there are a few methods for cpu evaluation and not all will work on any device. 
#### Distillation 
Distillation is performed using HuggingFace's `transformers` library, normally one would use a teacher and a student models that share tokenizers,
but we have managed to implement a custom distillation process that allows you to use models with different tokenizers for the process.
change `student_name` to distill into the model you want. the teacher model will be the one in `MODEL_DIR` make sure `model_name` matches.
#### Pruning
Pruning is performed using `torch` native pruning methods for unstructured pruning.
you can customize the amount of pruning by changing the `PRUNE_AMOUNT` variable, `PRUNE_EVAL` flag to enable/disable pruned model evaluation without retraining.
The model to be pruned is the model in `MODEL_DIR` make sure `model_name` matches.
##### counting
flag to enable/disable counting the number of parameters and disk size of the model provided in the `COUNTING_DIR` variable 
### Extras
#### HyperParameter Searching
A hyperparameter search script (`HPtrain.py`) is provided to find the best hyperparameters for training using the `Trainer` API.
Simply edit the hyperparameter search space at the top of the file along with the model_name and fast tokenization flag and run the script.
#### Label prediction procedure
We have devised a custom labeling prediction procedure (`procedure.py`) that incorporates a threshold mechanism and weighted confidence scores
to enhance the reliability of sentiment classification. This method seeks to be a better alternative to the default argmax approach.
usage of this file isn't as streamlined like the other scripts, you will need to look inside the functions in the file to see how to use it.
#### train diagnostics
this is a utility script (`train diagnostics.py`) that can be used to check weather or not training has happened or not by looking the norm of the weights in certain layers.

## Custom Training Pipeline

### Training

The custom training process uses **Weights & Biases (wandb)** for experiment tracking and **Optuna** for hyperparameter optimization.

#### Training Configuration
- **Models**: `cardiffnlp/twitter-roberta-base-sentiment` and `microsoft/deberta-v3-base`
- **Trials**: 15 hyperparameter optimization trials
- **Epochs**: 20 epochs per trial
- **Optimization**: Optuna with wandb integration

#### Running Training

1. Open `train.ipynb` in Jupyter
2. Set up your wandb credentials
3. Comment/uncomment the appropriate model and tokenizer sections for RoBERTa or DeBERTa
4. Run the notebook to start hyperparameter optimization

#### Output
Best models from each trial are saved in:
```
models/training/
├── roberta_best_model_trial_{trial_num}.pt
└── deberta_best_model_trial_{trial_num}.pt
```

### Compression

Three compression techniques are applied to the best models:

1. **Quantization** - Reduces model precision (FP32 → INT8/FP16)
2. **Pruning** - Removes less important weights
3. **Distillation** - Knowledge transfer to smaller models

#### Running Compression

Open `compress_roberta.ipynb` for RoBERTa or `compress_deberta.ipynb` for DeBERTa respectively.

#### Output Structure
```
models/{model_name}-full/
├── baseline/           # Original best model
├── quantized/          # Quantized model
├── pruned/            # Pruned model
├── distilled/         # Distilled model
├── comparison.csv     # Performance comparison
└── wandb_export_*.csv # Training metrics
```

### Comparison

Performance comparison is done by evaluating all compressed models on the evaluation dataset, computing metrics (accuracy, precision, recall, F1, AUC), and comparing results. Results are automatically generated and saved in `comparison.csv` files within each model directory.
