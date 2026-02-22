import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from data import prepare_data
from utils import compute_metrics, id2label, label2id

os.environ["WANDB_DISABLED"] = "true"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch


def _cuda_supported_for_current_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        return current_arch in supported_arches
    except Exception:
        return False

def train():
    # ---------------------------------------------------------
    # TASK 4 CODE: Select and load the model from Hugging Face
    # ---------------------------------------------------------
    model_name = 'distilbert-base-cased'
    requested_device = os.getenv("TRAIN_DEVICE", "auto").lower()
    cuda_usable = _cuda_supported_for_current_gpu()

    if requested_device == "cpu":
        device_name = 'cpu'
    elif requested_device == "cuda":
        if cuda_usable:
            device_name = 'cuda'
        else:
            print("TRAIN_DEVICE=cuda requested, but CUDA is not usable with this torch/GPU setup. Falling back to CPU.")
            device_name = 'cpu'
    else:
        device_name = 'cuda' if cuda_usable else 'cpu'
    cached_model_directory_name = 'distilbert-reviews-genres'

    print(f"Using device: {device_name}")

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    # Prepare the data using the tokenizer we just loaded
    train_dataset, test_dataset = prepare_data(tokenizer)

    print(f"Loading model {model_name}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label)
    )
    try:
        model = model.to(device_name)
    except RuntimeError as err:
        print(f"Device '{device_name}' failed ({err}). Falling back to CPU.")
        device_name = 'cpu'
        model = model.to(device_name)
    # ---------------------------------------------------------
    
    # TASK 5 CODE: Configure and run the Trainer API
    training_args = TrainingArguments(
        num_train_epochs=3,              
        per_device_train_batch_size=10,  
        per_device_eval_batch_size=16,   
        learning_rate=5e-5,              
        warmup_steps=100,                
        weight_decay=0.01,               
        output_dir='./results',          
        logging_dir='./logs',            
        logging_steps=100,               
        eval_strategy='steps',           
        no_cuda=(device_name == 'cpu'),
        report_to=[],  
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=test_dataset,           
        compute_metrics=compute_metrics      
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {cached_model_directory_name}...")
    trainer.save_model(cached_model_directory_name)
    tokenizer.save_pretrained(cached_model_directory_name)

if __name__ == "__main__":
    train()