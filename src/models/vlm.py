import torch
import torch.nn as nn
from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder

class MiniLLaVA(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(MiniLLaVA, self).__init__()
        self.device = device
        
        # Initialize the vision and text encoders.
        self.vision_encoder = VisionEncoder(device=device)
        self.text_encoder = TextEncoder(device=device)
        
        # Obtain the hidden size from the text encoder's configuration.
        # This ensures that our projection layer will output the correct dimension.
        text_hidden_dim = self.text_encoder.model.config.hidden_size  # Expected to be 2048 for microsoft/phi-1_5
        
        # Create a projection layer to map vision embeddings (512) into the text embedding space (text_hidden_dim).
        self.projection = nn.Linear(512, text_hidden_dim).to(self.device)
        
    def forward(self, image, text):
        # This forward method is not used directly in generative training.
        # Instead, the training script uses the vision_encoder, projection, and text_encoder separately.
        image_emb = self.vision_encoder.forward(image)
        projected_image_emb = self.projection(image_emb)
        text_logits = self.text_encoder.forward(text, prefix_embeds=projected_image_emb.unsqueeze(1))
        return text_logits

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        text_input = " ".join(sys.argv[2:])
        model = MiniLLaVA()
        # In inference mode, you can generate logits or use the generative model’s generate method.
        logits = model.text_encoder.forward(text_input)
        print("Logits Shape:", logits.shape)
    else:
        print("Usage: python src/models/vlm.py <image_path> <text>")
