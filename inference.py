import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def predict(text, model_path, threshold=0.5):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 512
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Decode labels (indices)
    predicted_labels = np.where(probs >= threshold)[0]
    # Map to 1-10 as per user requirement (assuming 1-based indexing for "1-10")
    predicted_labels = predicted_labels + 1
    
    return predicted_labels.tolist()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <text_file_or_string>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    input_text = sys.argv[2]
    
    if input_text.endswith(".txt"):
        with open(input_text, "r") as f:
            text = f.read()
    else:
        text = input_text
        
    labels = predict(text, model_path)
    print(f"Predicted Labels: {labels}")

