# src/models/vision_encoder.py

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(VisionEncoder, self).__init__()
        self.device = device
        # Load the CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()
    
    def forward(self, image_input):
        """
        Accepts either a file path (str) or a PIL image.
        Returns the image embeddings.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            # Assume it's already a PIL Image
            image = image_input.convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        encoder = VisionEncoder()
        embeddings = encoder.forward(image_path)
        print("Image Embeddings Shape:", embeddings.shape)
    else:
        print("Please provide the path to an image as a command-line argument.")
