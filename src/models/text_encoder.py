# src/models/text_encoder.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(TextEncoder, self).__init__()
        self.device = device
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode unless you plan to fine-tune

    def forward(self, text: str):
        """
        Process text input and return text embeddings.
        :param text: Input text string.
        :return: Text embeddings tensor.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # For DistilBERT, use the first token's embedding (analogous to the [CLS] token)
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        return text_embeddings

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
        encoder = TextEncoder()
        embeddings = encoder.forward(text_input)
        print("Text Embeddings Shape:", embeddings.shape)
    else:
        print("Please provide text as a command-line argument.")
