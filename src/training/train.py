#!/usr/bin/env python3
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

# Import LoRA functions from PEFT.
from peft import get_peft_model, LoraConfig

# Reuse the LLaVACaptionDataset.
from src.data.llava_caption_dataset import LLaVACaptionDataset
from src.models.vlm import MiniLLaVA  # Contains vision encoder, LLM, and projection.

#############################################
# Helper Functions
#############################################

def get_train_val_datasets(train_ratio=0.9, max_samples=10000, json_file=None, img_dir=None):
    dataset = LLaVACaptionDataset(json_file=json_file, img_dir=img_dir, split="train", max_samples=max_samples)
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    print(f"[train_llm] Dataset split: {train_size} train samples, {val_size} validation samples.")
    return random_split(dataset, [train_size, val_size])

def validate(model, val_loader, tokenizer, epoch, iteration, device, max_length):
    """
    For validation, generate text from the LM given a prompt and print shapes.
    """
    model.llm.model.eval()
    print(f"\n[Validation @ epoch {epoch}, iteration {iteration}]")
    for batch in val_loader:
        if len(batch) >= 4:
            images, prompts, answers, image_paths = batch[:4]
        else:
            continue
        for i in range(min(5, len(prompts))):
            # Here we only provide the prompt (with the "assistant:" marker) to generate the answer.
            prompt_text = f"user: {prompts[i].strip()}\nassistant:"
            with torch.no_grad():
                image_emb = model.vision_encoder(images[i])
                projected = model.projection(image_emb)
            generated_ids = model.llm.generate(
                [prompt_text],
                prefix_embeds=projected,  # add image features projected as prefix
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\n[Validation Generation] Sample:")
            print("Prompt Text:", prompt_text)
            print("Ground Truth:", answers[i].strip())
            print("Generated Text:", generated_text)
            print("Image Name:", os.path.basename(image_paths[i]))
            wandb.log({
                "val_prompt": prompt_text,
                "val_ground_truth": answers[i].strip(),
                "val_generated": generated_text,
                "image_name": os.path.basename(image_paths[i]),
                "epoch": epoch,
                "iteration": iteration,
            })
        break
    model.llm.model.train()

#############################################
# LM-only Fine-Tuning with LoRA (Next-Token Prediction)
#############################################

def train_llm():
    max_length = 128

    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",
        config={
            "lm_lr": 1e-4,
            "projector_lr": 1e-5,
            "batch_size": 4,
            "epochs": 10,
            "max_length": max_length,
            "description": "LM finetuning with visual prefix using LoRA, masking both prompt and image prefix.",
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Initialize the integrated model.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    # Freeze the vision encoder.
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Load pre-trained projector.
    pretrained_projector_path = "saved_models/projector_iter_102300_epoch31.pth"
    if os.path.exists(pretrained_projector_path):
        print(f"[train_llm] Loading pre-trained projector from {pretrained_projector_path}")
        model.projection.load_state_dict(torch.load(pretrained_projector_path, map_location=device))
    else:
        print("[train_llm] Pre-trained projector not found. Proceeding without loading.")

    # Inject LoRA adapters.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.llm.model = get_peft_model(model.llm.model, lora_config)
    model.llm.model.print_trainable_parameters()

    # Set up optimizer with separate parameter groups.
    optimizer = optim.AdamW([
        {"params": model.llm.model.parameters(), "lr": config.lm_lr},
        {"params": model.projection.parameters(), "lr": config.projector_lr}
    ])

    json_file = "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json"
    img_dir = "src/data/LLaVA_Instruct_Files/images"
    train_ds, val_ds = get_train_val_datasets(train_ratio=0.9, max_samples=10000, json_file=json_file, img_dir=img_dir)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True)

    tokenizer = model.llm.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    iteration = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        print(f"\n[train_llm] Starting Epoch {epoch+1}/{config.epochs}")
        for batch in train_loader:
            images, prompts, answers, _ = batch[:4]
            prefix_embeds = model.projection(model.vision_encoder.forward(images))
            # Instead of concatenating prompt and answer in one string,
            # we separate them so that we can mask out (ignore) the prompt tokens.
            prompt_texts = [f"user: {p.strip()}\nassistant:" for p in prompts]
            assistant_texts = [f" {a.strip()}" for a in answers]
            outputs = model.llm(prompt_texts, assistant_texts, prefix_embeds=prefix_embeds, max_length=max_length)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            print(f"[train_llm] Iteration {iteration}, Batch Loss: {loss.item():.4f}")

            # Save a checkpoint every 100 iterations.
            if iteration % 100 == 0:
                validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
                iter_save_path = os.path.join("saved_models", f"checkpoint_epoch{epoch+1}_iter.pth")
                checkpoint = {
                    "llm_state_dict": model.llm.state_dict(),
                    "projector_state_dict": model.projection.state_dict()
                }
                torch.save(checkpoint, iter_save_path)
                print(f"[train_llm] Updated checkpoint at iteration {iteration} for epoch {epoch+1} saved to {iter_save_path}")
            iteration += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[train_llm] Epoch {epoch+1} complete. Avg Epoch Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "epoch_loss": avg_epoch_loss})
        epoch_save_path = os.path.join("saved_models", f"checkpoint_epoch{epoch+1}.pth")
        os.makedirs("saved_models", exist_ok=True)
        checkpoint = {
            "llm_state_dict": model.llm.state_dict(),
            "projector_state_dict": model.projection.state_dict()
        }
        torch.save(checkpoint, epoch_save_path)
        print(f"[train_llm] Saved end-of-epoch checkpoint for epoch {epoch+1} to {epoch_save_path}")
        torch.cuda.empty_cache()
        validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
        model.llm.model.train()

        total_norm_proj = sum(p.grad.data.norm(2).item() for p in model.projection.parameters() if p.grad is not None)
        total_norm_lm = sum(p.grad.data.norm(2).item() for p in model.llm.model.parameters() if p.grad is not None)
        print(f"[train_llm] Gradient Norms - Projector: {total_norm_proj:.4f}, LM: {total_norm_lm:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train_llm()
