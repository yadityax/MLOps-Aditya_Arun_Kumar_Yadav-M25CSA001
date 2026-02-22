import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer
from data import prepare_data
from utils import compute_metrics, id2label
from sklearn.metrics import classification_report

def evaluate_from_hub():
    # Replace 'your-username' with your actual Hugging Face username
    hf_repo_name = 'm25csa001/distilbert-reviews-genres'

    print(f"Downloading tokenizer and dataset from Hub: {hf_repo_name}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(hf_repo_name)
    
    # Prepare the test dataset
    _, test_dataset = prepare_data(tokenizer)

    print(f"Downloading model from Hub: {hf_repo_name}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        hf_repo_name, num_labels=len(id2label)
    )

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    print("Running evaluation on the downloaded model...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print("\nHub Model Evaluation Results:", eval_results)
    
    # Generate F1, Precision, and Recall
    print("\nGenerating classification report...")
    predicted_results = trainer.predict(test_dataset)
    predicted_labels = predicted_results.predictions.argmax(-1)
    
    y_true = [id2label[l] for l in test_dataset.labels]
    y_pred = [id2label[l] for l in predicted_labels.flatten().tolist()]
    
    report = classification_report(y_true, y_pred)
    print(report)

if __name__ == "__main__":
    evaluate_from_hub()