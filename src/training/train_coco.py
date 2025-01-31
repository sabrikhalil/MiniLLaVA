# src/training/train_coco.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vlm import MiniLLaVA  # This is our integrated model (vision + text encoders + projection)
from data.coco_dataset import CocoCaptionsDataset
import wandb

def train():
    # Initialize wandb with your project configuration.
    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 3,  # For demonstration; increase as needed.
            "loss_function": "MSELoss",
            "optimizer": "Adam",
            "description": "Pre-training stage: finetuning only the projection layer using COCO Captions dataset."
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the MiniLLaVA model.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    # Freeze the vision and text encoders.
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

    # Load the COCO Captions dataset (limit max_samples if desired for testing).
    dataset = CocoCaptionsDataset(split="train", max_samples=1000)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Training loop.
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for images, texts in dataloader:
            optimizer.zero_grad()
            batch_loss = 0.0

            # Process each sample in the batch.
            # (For simplicity, we process each sample individually; you could adapt this to batch processing.)
            for image, text in zip(images, texts):
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
