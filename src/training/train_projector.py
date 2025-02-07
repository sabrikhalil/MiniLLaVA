#!/usr/bin/env python3
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from torch.cuda.amp import autocast, GradScaler

# Import LoRA functions from PEFT.
from peft import get_peft_model, LoraConfig

# Reuse the LLaVACaptionDataset (which now returns a 4-tuple:
# (image, user_prompt, assistant_answer, image_path)).
from src.data.llava_caption_dataset import LLaVACaptionDataset
from src.models.vlm import MiniLLaVA  # MiniLLaVA includes vision encoder, LLM, and projection layer.

#############################################
# Helper Functions
#############################################

def get_train_val_datasets(train_ratio=0.9, max_samples=10000, json_file=None, img_dir=None):
    dataset = LLaVACaptionDataset(json_file=json_file, img_dir=img_dir, split="train", max_samples=max_samples)
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    print(f"Dataset split: {train_size} train samples, {val_size} validation samples.")
    return random_split(dataset, [train_size, val_size])

def validate(model, val_loader, tokenizer, epoch, iteration, device, max_length):
    """
    For validation, run a generation evaluation using the current projector.
    """
    model.llm.model.eval()
    print(f"\n[Validation @ epoch {epoch}, iteration {iteration}]")
    for batch in val_loader:
        if len(batch) >= 4:
            images, prompts, answers, image_paths = batch[:4]
        else:
            continue
        for i in range(min(5, len(prompts))):
            prompt_text = f"user: {prompts[i].strip()}\nassistant:"
            # Compute prefix for this image.
            with torch.no_grad():
                image_emb = model.vision_encoder(images[i])
                projected = model.projection(image_emb)
            generated_ids = model.llm.generate(
                [prompt_text],
                prefix_embeds=projected,
                max_new_tokens=25,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\n=== Generation Eval Sample ===")
            print("Input Text:", prompt_text)
            print("Ground Truth Assistant Answer:", answers[i].strip())
            print("Generated Assistant Answer:", generated_text)
            print("Image Name:", os.path.basename(image_paths[i]))
            wandb.log({
                "val_prompt_gen": prompt_text,
                "val_ground_truth_gen": answers[i].strip(),
                "val_generated_gen": generated_text,
                "image_name_gen": os.path.basename(image_paths[i]),
                "epoch": epoch,
                "iteration": iteration,
            })
        break
    model.llm.model.train()

#############################################
# Train Projector (with Frozen Vision Encoder & LLM)
#############################################

def train_projector():
    max_length = 64

    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 12,
            "epochs": 10,
            "max_length": max_length,
            "description": "Stage 1: Fine-tuning only the image projector for feature alignment. Vision encoder and LLM are frozen.",
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Initialize the integrated model.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    # Freeze the vision encoder and LLM.
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.llm.model.parameters():
        param.requires_grad = False

    # (The projector is not frozen and will be fine-tuned.)
    optimizer = optim.AdamW(model.projection.parameters(), lr=config.learning_rate)
    scaler = GradScaler()

    # Get training and validation datasets.
    json_file = "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json"
    img_dir = "src/data/LLaVA_Instruct_Files/images"
    train_ds, val_ds = get_train_val_datasets(train_ratio=0.9, max_samples=10000,
                                               json_file=json_file, img_dir=img_dir)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True)

    tokenizer = model.llm.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    iteration = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        print(f"\n=== Starting Epoch {epoch+1}/{config.epochs} ===")
        for batch in train_loader:
            images, prompts, answers, _ = batch[:4]
            # Compute the visual prefix embeddings.
            prefix_embeds = model.projection(model.vision_encoder.forward(images))
            # Construct separate lists for prompt and assistant texts.
            prompt_texts = [f"user: {p.strip()}\nassistant:" for p in prompts]
            assistant_texts = [f" {a.strip()}" for a in answers]
            
            optimizer.zero_grad()
            with autocast():
                # Forward pass using the updated LM forward signature.
                outputs = model.llm(prompt_texts, assistant_texts, prefix_embeds=prefix_embeds, max_length=max_length)
                loss = outputs.loss
            scaler.scale(loss).backward()
            
            # Clip projector gradients and log their norm.
            torch.nn.utils.clip_grad_norm_(model.projection.parameters(), max_norm=1.0)
            total_norm = sum(p.grad.data.norm(2).item() for p in model.projection.parameters() if p.grad is not None)
            print(f"Projection grad norm: {total_norm:.4f}")
            wandb.log({"proj_grad_norm": total_norm})
            
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            print(f"Iteration {iteration}, Batch Loss: {loss.item():.4f}")

            if iteration % 100 == 0:
                iter_save_path = os.path.join("saved_models", f"projector_iter_{iteration}_epoch{epoch+1}.pth")
                os.makedirs("saved_models", exist_ok=True)
                torch.save(model.projection.state_dict(), iter_save_path)
                print(f"Saved projection weights at iteration {iteration} (epoch {epoch+1}) to {iter_save_path}")
                validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
            iteration += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Epoch Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "epoch_loss": avg_epoch_loss})
        epoch_save_path = os.path.join("saved_models", f"projector_epoch{epoch+1}.pth")
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.projection.state_dict(), epoch_save_path)
        print(f"Saved projection weights at end of epoch {epoch+1} to {epoch_save_path}")
        torch.cuda.empty_cache()
        validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
        model.llm.model.train()

    wandb.finish()

if __name__ == "__main__":
    train_projector()
