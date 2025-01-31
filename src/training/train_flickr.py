# src/training/train_flickr.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vlm import MiniLLaVA  # Our integrated model (vision encoder, text encoder, and projection layer)
from data.caption_dataset import CaptionDataset
import wandb

def train():
    # Initialize wandb with project settings.
    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 3,  # Adjust epochs as needed.
            "loss_function": "MSELoss",
            "optimizer": "Adam",
            "description": "Pre-training stage: finetuning the projection layer (fixed vision and text encoders) on the Flickr30k dataset."
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the MiniLLaVA model.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    # Freeze vision and text encoder parameters.
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    # Only the projection layer is trainable.
    for param in model.projection.parameters():
        param.requires_grad = True

    # Set up the optimizer for the projection layer.
    optimizer = optim.Adam(model.projection.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Load the Flickr30k dataset using our CaptionDataset class.
    dataset = CaptionDataset(split="train", max_samples=1000)  # You can remove max_samples for full dataset.
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Training loop.
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for images, texts in dataloader:
            optimizer.zero_grad()
            batch_loss = 0.0

            # Process each sample in the batch.
            for image, text in zip(images, texts):
                # Our model now accepts a PIL image (or file path) and text.
                projected_image_emb, text_emb = model.forward(image, text)
                loss = criterion(projected_image_emb, text_emb)
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            avg_batch_loss = batch_loss / len(images)
            epoch_loss += avg_batch_loss
            wandb.log({"batch_loss": avg_batch_loss})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "epoch_loss": avg_epoch_loss})

    wandb.finish()

if __name__ == "__main__":
    train()
