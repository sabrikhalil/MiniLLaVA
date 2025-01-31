# src/data/coco_dataset.py

from torch.utils.data import Dataset
from datasets import load_dataset, Image as DatasetsImage

class CocoCaptionsDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        """
        Loads the COCO Captions dataset (2017 version) using Hugging Face's datasets.
        
        Args:
            split (str): Which split to load ("train", "validation", etc.).
            max_samples (int, optional): If provided, limits the number of samples (useful for debugging).
        """
        # Load the dataset from Hugging Face
        self.dataset = load_dataset("coco_captions", "2017", split=split)
        # Cast the "image" column to the Image type so that images are automatically decoded as PIL images.
        self.dataset = self.dataset.cast_column("image", DatasetsImage())
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # The "image" column now contains a PIL image.
        image = sample["image"]
        # The COCO Captions dataset typically returns a list of captions; we take the first one.
        caption = sample["captions"][0] if isinstance(sample["captions"], list) else sample["captions"]
        return image, caption
