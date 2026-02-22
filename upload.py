from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from huggingface_hub import login


login(token="<YOUR_HUGGINGFACE_TOKEN>")


hf_username = "m25csa001"
repo_name = f"{hf_username}/distilbert-reviews-genres"
model_dir = "distilbert-reviews-genres" # The folder where we saved the model in Task 5

print("Loading locally saved model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)


print(f"Pushing model and tokenizer to Hugging Face Hub: {repo_name}...")


tokenizer.push_to_hub(repo_name)
model.push_to_hub(repo_name)

print(f"Upload complete! Check your profile at: https://huggingface.co/{repo_name}")