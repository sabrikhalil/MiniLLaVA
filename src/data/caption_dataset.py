from datasets import load_dataset, Image as DatasetsImage
from torch.utils.data import Dataset
from torchvision import transforms

class CaptionDataset(Dataset):
    def __init__(self, split="train", max_samples=None, transform=None):
        """
        Loads the nlphuji/flickr30k dataset using Hugging Face's datasets library.
        Assumes the dataset contains an "image" column (a PIL image) and a 
        "caption" field containing the caption(s). Applies a transform to resize
        images to 224x224 and convert them to tensors.
        
        Args:
            split (str): Which split to load ("train", "test", etc.).
            max_samples (int, optional): Limit the number of samples for debugging.
            transform (callable, optional): A transformation to apply to the PIL image.
                Defaults to resizing to 224x224 and converting to tensor.
        """
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        self.dataset = self.dataset.cast_column("image", DatasetsImage())
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))
        
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Use the "image" column (decoded to PIL) and caption from "caption"
        image = sample["image"]
        caption = sample["caption"][0] if isinstance(sample["caption"], list) else sample["caption"]
        image = self.transform(image)
        return image, caption

if __name__ == "__main__":
    dataset = CaptionDataset(split="test", max_samples=5)
    print(f"Dataset size: {len(dataset)}")
    for idx in range(len(dataset)):
        img, cap = dataset[idx]
        print(f"Sample {idx}: Caption: {cap}, Image tensor shape: {img.shape}")
