# src/models/vision_encoder.py

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(VisionEncoder, self).__init__()
        self.device = device
        # Load the CLIP model (we use the full model, but will use the vision encoder part)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()
    
    def forward(self, image_path: str):
        """
        Process an image and return the image embeddings.
        :param image_path: Path to the input image.
        :return: Image embeddings tensor.
        """
        # Open the image and ensure it is in RGB mode
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract image features using the vision encoder
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features

if __name__ == "__main__":
    # Quick test: run this module with an image path to see the output embeddings.
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        encoder = VisionEncoder()
        embeddings = encoder.forward(image_path)
        print("Image Embeddings Shape:", embeddings.shape)
    else:
        print("Please provide the path to an image as a command-line argument.")
