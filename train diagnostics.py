from transformers import AutoModelForSequenceClassification
import torch

# Load the original pre-trained model
original_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment",
    num_labels=5,
    ignore_mismatched_sizes=True
)

# Load your fine-tuned model
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
    "trainer/run-14/checkpoint-7216"  # Path to your saved model
)

# Compare weights from different parts of the model
def compare_model_weights(model1, model2):
    results = {}

    # Check early, middle and late transformer layers
    first_layer = model1.roberta.encoder.layer[0].attention.self.query.weight
    first_layer_ft = model2.roberta.encoder.layer[0].attention.self.query.weight
    first_layer_diff = torch.norm(first_layer - first_layer_ft).item()

    mid_idx = len(model1.roberta.encoder.layer) // 2
    mid_layer = model1.roberta.encoder.layer[mid_idx].attention.self.query.weight
    mid_layer_ft = model2.roberta.encoder.layer[mid_idx].attention.self.query.weight
    mid_layer_diff = torch.norm(mid_layer - mid_layer_ft).item()

    last_idx = len(model1.roberta.encoder.layer) - 1
    last_layer = model1.roberta.encoder.layer[last_idx].attention.self.query.weight
    last_layer_ft = model2.roberta.encoder.layer[last_idx].attention.self.query.weight
    last_layer_diff = torch.norm(last_layer - last_layer_ft).item()

    # Check classification head
    classifier = model1.classifier.dense.weight
    classifier_ft = model2.classifier.dense.weight
    classifier_diff = torch.norm(classifier - classifier_ft).item()

    return {
        'first_layer_diff': first_layer_diff,
        'mid_layer_diff': mid_layer_diff,
        'last_layer_diff': last_layer_diff,
        'classifier_diff': classifier_diff
    }

differences = compare_model_weights(original_model, fine_tuned_model)

print("Weight differences between original and fine-tuned model:")
for layer, diff in differences.items():
    print(f"{layer}: {diff:.6f}")
    status = "UPDATED" if diff > 0.01 else "NOT UPDATED (frozen)"
    print(f"  → {status}")