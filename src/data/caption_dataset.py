# src/data/caption_dataset.py

from torch.utils.data import Dataset
from datasets import load_dataset, Image as DatasetsImage

class CaptionDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        """
        Loads the Flickr30k dataset using Hugging Face's datasets library.
        
        Args:
            split (str): Which split to load ("train", "validation", etc.).
            max_samples (int, optional): If provided, limits the number of samples (useful for debugging).
        """
        # Load the dataset from Hugging Face; here we use "flickr30k"
        self.dataset = load_dataset("flickr30k", split=split)
        # Ensure the "image" column is decoded as PIL images.
        self.dataset = self.dataset.cast_column("image", DatasetsImage())
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # "image" column now contains a PIL image.
        image = sample["image"]
        # The flickr30k dataset typically returns a list of sentences in the "sentence" field.
        # We take the first sentence.
        caption = sample["sentence"][0] if isinstance(sample["sentence"], list) else sample["sentence"]
        return image, caption

# For quick testing, run this file:
if __name__ == "__main__":
    dataset = CaptionDataset(split="train", max_samples=5)
    for idx in range(len(dataset)):
        image, caption = dataset[idx]
        print(f"Sample {idx}: Caption: {caption}, Image type: {type(image)}")
