import torch
import torch.nn as nn
from src.models.vision_encoder import VisionEncoder
from src.models.llm import LLM

class Projection(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, text_dim * 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(text_dim * 2, text_dim)
        # Initialize weights similar to LLaVA's projection.
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MiniLLaVA(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(MiniLLaVA, self).__init__()
        self.device = device
        
        self.vision_encoder = VisionEncoder(device=device)
        self.llm = LLM(device=device)  # renamed and refactored LLM
        
        # Get dimensions from the encoders.
        vision_dim = self.vision_encoder.output_dim  # Should match your vision encoder's output dimension.
        text_hidden_dim = self.llm.model.config.hidden_size
        
        self.projection = Projection(vision_dim, text_hidden_dim).to(self.device)
        
        # Add special tokens to the tokenizer.
        tokenizer = self.llm.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG_START]", "[IMG_END]"]})
        self.llm.model.resize_token_embeddings(len(tokenizer))
        
    def forward(self, image, text):
        """
        Given an image and text prompt, extract visual features, project them,
        and forward them as a prefix to the language model.
        """
        image_emb = self.vision_encoder.forward(image)
        projected_image_emb = self.projection(image_emb)
        # Unsqueeze to create a prefix of shape (batch_size, 1, embed_dim).
        text_logits = self.llm.forward(text, prefix_embeds=projected_image_emb.unsqueeze(1))
        return text_logits

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        text_input = " ".join(sys.argv[2:])
        model = MiniLLaVA()
        # For inference, use the generate method of the LLM.
        generated_ids = model.llm.generate(
            [text_input],
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            pad_token_id=model.llm.tokenizer.eos_token_id
        )
        generated_text = model.llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Generated Text:", generated_text)
    else:
        print("Usage: python src/models/vlm.py <image_path> <text>")
