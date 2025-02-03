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
    
    def forward(self, image_input):
        """
        Accepts image_input as either:
          - a tensor (shape [3, 224, 224]) or a batch of images (shape [B, 3, 224, 224]),
          - a filepath, or a PIL Image.
        Converts to PIL image(s) if necessary and returns image features.
        """
        # If input is a torch.Tensor, check its dimensions.
        if isinstance(image_input, torch.Tensor):
            # If it's a batch of images (4D), convert each image separately.
            if image_input.ndim == 4:
                to_pil = transforms.ToPILImage()
                images = [to_pil(img.cpu()) for img in image_input]
            elif image_input.ndim == 3:
                to_pil = transforms.ToPILImage()
                images = to_pil(image_input.cpu())
            else:
                raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {image_input.ndim} dimensions.")
        elif isinstance(image_input, str):
            # If a file path is provided, open the image.
            images = Image.open(image_input).convert("RGB")
        else:
            # Assume input is a PIL Image or list of PIL Images.
            images = image_input
        
        # Use the CLIP processor. The processor accepts a single PIL image or a list of them.
        inputs = self.processor(images=images, return_tensors="pt")
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
