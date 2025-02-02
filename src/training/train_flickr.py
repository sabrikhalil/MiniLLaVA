import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from torch.nn.functional import cosine_similarity
from src.models.vlm import MiniLLaVA
from src.data.caption_dataset import CaptionDataset

def get_train_val_datasets(train_ratio=0.9, max_samples=10000):
    full_dataset = CaptionDataset(split="test", max_samples=max_samples)
    total = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")
    return train_dataset, val_dataset

def train():
    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": 3,
            "loss_function": "MSELoss",
            "optimizer": "Adam",
            "description": "Finetuning the projection layer on nlphuji/flickr30k (10K samples, 90/10 split)."
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize integrated model and freeze vision and text encoders.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.projection.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.projection.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_dataset, val_dataset = get_train_val_datasets(train_ratio=0.9, max_samples=10000)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    global_iteration = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        print(f"\nStarting Epoch {epoch+1}/{config.epochs}")
        for i, (images, texts) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_samples = 0

            # Log details for the first batch of the first epoch.
            if epoch == 0 and i == 0:
                print("First batch sample image tensor shapes:")
                for img in images:
                    print(img.shape)
                print("First sample caption:", texts[0])
                wandb.log({"first_batch_image_shapes": [img.shape for img in images],
                           "first_sample_caption": texts[0]})

            for image, text in zip(images, texts):
                if isinstance(text, list):
                    text = text[0]
                proj_img_emb, txt_emb = model.forward(image, text)
                loss = criterion(proj_img_emb, txt_emb)
                loss.backward()
                batch_loss += loss.item()
                valid_samples += 1
                global_iteration += 1
                if global_iteration % 10 == 0:
                    wandb.log({"iteration": global_iteration, "current_loss": loss.item()})
            if valid_samples > 0:
                optimizer.step()
                avg_batch_loss = batch_loss / valid_samples
            else:
                avg_batch_loss = 0.0
            epoch_loss += avg_batch_loss
            print(f"Epoch {epoch+1}, Batch {i+1}: Avg Loss = {avg_batch_loss:.4f} over {valid_samples} samples")
            wandb.log({"epoch": epoch+1, "batch": i+1, "batch_loss": avg_batch_loss})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Epoch Loss = {avg_epoch_loss:.4f}")
        wandb.log({"epoch_loss": avg_epoch_loss})
        
        # Validation: process a few samples and log cosine similarities.
        model.eval()
        cos_sims = []
        val_samples = 0
        for images, texts in val_loader:
            for image, text in zip(images, texts):
                if isinstance(text, list):
                    text = text[0]
                proj_img_emb, txt_emb = model.forward(image, text)
                sim = cosine_similarity(proj_img_emb, txt_emb, dim=1)
                # If sim is a batch (more than one element), take the mean.
                sim_val = sim.mean().item() if sim.numel() > 1 else sim.item()
                cos_sims.append(sim_val)
                val_samples += 1
                if val_samples in {1, 10, 20}:
                    print(f"Validation sample {val_samples}: Caption: {text[:75]}..., Cosine Similarity: {sim_val:.4f}")
            if val_samples >= 20:
                break
        avg_cos_sim = np.mean(cos_sims) if cos_sims else 0.0
        wandb.log({"validation_avg_cosine_similarity": avg_cos_sim})
        model.train()
    wandb.finish()

if __name__ == "__main__":
    train()
