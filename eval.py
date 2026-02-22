import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer
from data import prepare_data
from utils import compute_metrics, id2label
from sklearn.metrics import classification_report

def evaluate():
    model_name = 'distilbert-base-cased'
    cached_model_directory_name = 'distilbert-reviews-genres'

    print("Loading tokenizer and dataset...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(cached_model_directory_name)
    
    # We only need the test dataset for evaluation
    _, test_dataset = prepare_data(tokenizer)

    print(f"Loading local saved model from {cached_model_directory_name}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        cached_model_directory_name, num_labels=len(id2label)
    )

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    print("Running evaluation...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print("\nEvaluation Results:", eval_results)
    
    # Save the basic Hugging Face evaluation metrics (Accuracy, Loss, etc.)
    with open("evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    print("Saved basic evaluation results to evaluation_results.json")
    
    # Generate detailed F1, Precision, and Recall using scikit-learn
    print("\nGenerating classification report...")
    predicted_results = trainer.predict(test_dataset)
    predicted_labels = predicted_results.predictions.argmax(-1)
    
    # FIX: Removed the .item() call here because test_dataset.labels are already standard integers
    y_true = [id2label[l] for l in test_dataset.labels]
    y_pred = [id2label[l] for l in predicted_labels.flatten().tolist()]
    
    report = classification_report(y_true, y_pred)
    print(report)

    # Save the detailed classification report
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("Saved detailed classification report to classification_report.txt")

if __name__ == "__main__":
    evaluate()