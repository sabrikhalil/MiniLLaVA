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
        print(f"[Projection.__init__] Initialized with vision_dim={vision_dim}, text_dim={text_dim}")

    def forward(self, x):
        out = self.fc2(self.gelu(self.fc1(x)))
        return out

class MiniLLaVA(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(MiniLLaVA, self).__init__()
        self.device = device
        
        # Initialize vision encoder and LLM.
        self.vision_encoder = VisionEncoder(device=device)
        self.llm = LLM(device=device)
        
        # Get dimensions from the encoders.
        vision_dim = self.vision_encoder.output_dim  # e.g. 768 from CLIP.
        text_hidden_dim = self.llm.model.config.hidden_size
        print(f"[MiniLLaVA.__init__] vision_dim={vision_dim}, text_hidden_dim={text_hidden_dim}")
        
        self.projection = Projection(vision_dim, text_hidden_dim).to(self.device)
        
        # Add special tokens to the tokenizer.
        tokenizer = self.llm.tokenizer
        special_tokens_dict = {"additional_special_tokens": ["[IMG_START]", "[IMG_END]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"[MiniLLaVA.__init__] Added {num_added_toks} special tokens: {special_tokens_dict['additional_special_tokens']}")
        self.llm.model.resize_token_embeddings(len(tokenizer))
        print(f"[MiniLLaVA.__init__] Tokenizer resized. New vocab size: {len(tokenizer)}")
        
        # Optionally, store the special token IDs for later use.
        self.img_start_id = tokenizer.convert_tokens_to_ids("[IMG_START]")
        self.img_end_id = tokenizer.convert_tokens_to_ids("[IMG_END]")
        print(f"[MiniLLaVA.__init__] [IMG_START] id: {self.img_start_id}, [IMG_END] id: {self.img_end_id}")

    def forward(self, image, text):
        """
        Given an image and text prompt, extract visual features, project them,
        and forward them as a prefix to the language model.
        """
        print("[MiniLLaVA.forward] Starting forward pass...")
        image_emb = self.vision_encoder.forward(image)
        print("[MiniLLaVA.forward] Image embeddings shape:", image_emb.shape)
        projected_image_emb = self.projection(image_emb)
        print("[MiniLLaVA.forward] Projected image embeddings shape:", projected_image_emb.shape)
        # Here, you would typically wrap your projected embeddings with [IMG_START] and [IMG_END] tokens.
        # For instance, using the special token IDs (this step could be done in the LLM class as well).
        # In this example, we'll assume that the LLM's forward method will handle the visual prefix.
        # Unsqueeze to create a prefix of shape (batch_size, 1, embed_dim) if needed.
        prefix_embeds = projected_image_emb.unsqueeze(1)
        print("[MiniLLaVA.forward] Prefix embeddings shape (after unsqueeze):", prefix_embeds.shape)
        text_logits = self.llm.forward(text, prefix_embeds=prefix_embeds)
        print("[MiniLLaVA.forward] Text logits shape:", text_logits.logits.shape)
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
            prefix_embeds=model.projection(model.vision_encoder.forward(image_path)).unsqueeze(1),
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            pad_token_id=model.llm.tokenizer.eos_token_id
        )
        generated_text = model.llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Generated Text:", generated_text)
    else:
        print("Usage: python src/models/vlm.py <image_path> <text>")
