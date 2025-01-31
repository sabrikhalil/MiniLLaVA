# src/models/vlm.py

import torch
import torch.nn as nn
from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder

class MiniLLaVA(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(MiniLLaVA, self).__init__()
        self.device = device
        
        # Initialize the vision and text encoders
        self.vision_encoder = VisionEncoder(device=device)
        self.text_encoder = TextEncoder(device=device)
        
        # Assume VisionEncoder outputs 512-dimensional embeddings and TextEncoder outputs 768-dimensional embeddings.
        # Create a projection layer to map vision embeddings into the text embedding space.
        self.projection = nn.Linear(512, 768).to(self.device)
        
    def forward(self, image_path: str, text: str):
        # Get image embeddings from the vision encoder
        image_emb = self.vision_encoder.forward(image_path)
        # Project image embeddings to match text embedding dimension
        projected_image_emb = self.projection(image_emb)
        
        # Get text embeddings from the text encoder
        text_emb = self.text_encoder.forward(text)
        
        return projected_image_emb, text_emb
    
    def compute_similarity(self, image_path: str, text: str):
        """
        Computes cosine similarity between projected image embeddings and text embeddings.
        """
        proj_image_emb, text_emb = self.forward(image_path, text)
        # Normalize embeddings
        proj_image_norm = proj_image_emb / proj_image_emb.norm(dim=1, keepdim=True)
        text_norm = text_emb / text_emb.norm(dim=1, keepdim=True)
        # Compute cosine similarity
        cosine_sim = torch.mm(proj_image_norm, text_norm.transpose(0, 1))
        return cosine_sim

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        text_input = " ".join(sys.argv[2:])
        model = MiniLLaVA()
        similarity = model.compute_similarity(image_path, text_input)
        print("Cosine Similarity:", similarity.item())
    else:
        print("Usage: python src/models/vlm.py <image_path> <text>")
