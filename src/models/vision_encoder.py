import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms

class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(VisionEncoder, self).__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self.output_dim = 768
    
    def forward(self, image_input):
        """
        Accepts image_input as either:
          - a tensor (shape [3, 224, 224]) or a batch of images (shape [B, 3, 224, 224]),
          - a filepath, or a PIL Image.
        Converts to PIL image(s) if necessary and returns patch tokens.

        Returns:
            patch_embeddings (torch.Tensor): A tensor of shape (B, n_patches, embed_dim) where:
                - B is the batch size.
                - n_patches is the number of image patches (typically, the total tokens minus one for the CLS token).
                - embed_dim is the CLIP vision model embedding dimension.
        """
        # Convert input to PIL image(s) if needed.
        if isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                to_pil = transforms.ToPILImage()
                images = [to_pil(img.cpu()) for img in image_input]
            elif image_input.ndim == 3:
                to_pil = transforms.ToPILImage()
                images = to_pil(image_input.cpu())
            else:
                raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {image_input.ndim} dimensions.")
        elif isinstance(image_input, str):
            images = Image.open(image_input).convert("RGB")
        else:
            images = image_input  # Assume it's a PIL Image or list thereof.
        
        # Process the image(s) to get model inputs.
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass through the vision model to get hidden states.
        # We set output_hidden_states=True to obtain all token embeddings.
        with torch.no_grad():
            outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
            # outputs.hidden_states[-1] is of shape (B, num_tokens, embed_dim).
            # Typically, the first token is the CLS token, so we remove it to obtain only patch tokens.
            patch_embeddings = outputs.hidden_states[-1][:, 0:, :] ## no need to discard the CLS token


        return patch_embeddings

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        encoder = VisionEncoder()
        embeddings = encoder.forward(image_path)
        print("Image Embeddings Shape:", embeddings.shape)
    else:
        print("Please provide the path to an image as a command-line argument.")
