import os
from datasets import load_dataset  # Do not cast to Image feature here since we want the raw file path.
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LLaVACaptionDataset(Dataset):
    def __init__(self, json_file, img_dir, split="train", max_samples=None, transform=None):
        """
        Loads the LLaVA-Instruct-150k dataset from a JSON file and images from a directory.
        Constructs training samples as triplets: (image, user_prompt, assistant_answer).
        Only retains rows where the corresponding image file exists.
        
        Args:
            json_file (str): Path to the JSON file (e.g., "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json").
            img_dir (str): Directory where the image files are stored (e.g., "src/data/LLaVA_Instruct_Files/images").
            split (str): Which split to load ("train", "test", etc.).
            max_samples (int, optional): Limit number of records for debugging.
            transform (callable, optional): Image transform to apply.
                Defaults to resizing to 224x224 and converting to a tensor.
        """
        self.img_dir = img_dir

        # Load JSON file as a Hugging Face dataset.
        dataset = load_dataset("json", data_files=json_file, split=split)
        if max_samples is not None:
            dataset = dataset.select(range(max_samples))
        
        # Update each example to include the full image path.
        def update_image_path(example):
            path = example.get("image")
            if isinstance(path, str) and path.strip():
                example["image"] = os.path.join(img_dir, path.strip())
            return example

        dataset = dataset.map(update_image_path)
        
        # Filter out examples where the image file does not exist.
        def has_image(example):
            path = example.get("image")
            return path is not None and os.path.exists(path)
        
        print("Filtering dataset to keep only examples with existing images...")
        dataset = dataset.filter(has_image)
        
        # Build training samples:
        # For every record, look at the "conversations" list.
        # Whenever a "human" turn is immediately followed by a "gpt" turn, create a sample.
        self.samples = []
        for record in dataset:
            conv = record.get("conversations", [])
            if not conv or len(conv) < 2:
                continue
            for i in range(len(conv) - 1):
                if conv[i].get("from", "").strip().lower() == "human" and conv[i+1].get("from", "").strip().lower() == "gpt":
                    user_text = conv[i].get("value", "").replace("<image>", "").strip()
                    assistant_text = conv[i+1].get("value", "").strip()
                    if user_text and assistant_text:
                        self.samples.append({
                            "image": record["image"],  # Full file path.
                            "user_prompt": user_text,
                            "assistant_answer": assistant_text
                        })
                        
        print(f"Constructed {len(self.samples)} training samples from {len(dataset)} records.")
        
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_field = sample["image"]
        # If img_field is a string, open it; if it's already an Image, use it directly.
        if isinstance(img_field, str):
            try:
                image = Image.open(img_field).convert("RGB")
            except Exception as e:
                raise FileNotFoundError(f"Could not open image at {img_field}: {e}")
        elif isinstance(img_field, Image.Image):
            image = img_field
        else:
            raise ValueError("Unexpected type for image field.")
        if self.transform:
            image = self.transform(image)
        user_prompt = sample["user_prompt"]
        assistant_answer = sample["assistant_answer"]
        # Return a tuple: (image, user_prompt, assistant_answer)
        return image, user_prompt, assistant_answer

if __name__ == "__main__":
    # Update these paths to your local environment.
    json_file = "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json"
    img_dir = "src/data/LLaVA_Instruct_Files/images"
    
    dataset = LLaVACaptionDataset(json_file, img_dir, split="train", max_samples=500)
    print(f"Total training samples: {len(dataset)}")
    for idx in range(min(len(dataset), 5)):
        img, user_prompt, assistant_answer = dataset[idx]
        print(f"Sample {idx}:")
        print("User Prompt:", user_prompt)
        print("Assistant Answer:", assistant_answer)
        print("Image tensor shape:", img.shape)
