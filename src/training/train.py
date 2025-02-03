import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import wandb
from src.models.vlm import MiniLLaVA
from src.data.llava_caption_dataset import LLaVACaptionDataset

def get_train_val_datasets(train_ratio=0.9, max_samples=10000, json_file=None, img_dir=None):
    # Use the LLaVACaptionDataset in place of the old CaptionDataset.
    full_dataset = LLaVACaptionDataset(json_file=json_file, img_dir=img_dir, split="train", max_samples=max_samples)
    total = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")
    return train_dataset, val_dataset

def train():
    # Initialize Weights & Biases for logging.
    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",  # replace with your W&B entity
        config={
            "learning_rate": 1e-4,
            "batch_size": 2,  # adjust according to your GPU memory
            "epochs": 3,
            "description": "Generative finetuning: conditioning an LLM with image prefix embeddings.",
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the integrated model (MiniLLaVA) and freeze vision and text encoder parameters.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.projection.parameters():
        param.requires_grad = True  # Only train the projection layer.

    optimizer = optim.Adam(model.projection.parameters(), lr=config.learning_rate)

    # Provide the correct paths for your dataset.
    json_file = "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json"
    img_dir = "src/data/LLaVA_Instruct_Files/images"
    train_dataset, val_dataset = get_train_val_datasets(train_ratio=0.9, max_samples=10000,
                                                         json_file=json_file, img_dir=img_dir)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Get the tokenizer from the text encoder.
    tokenizer = model.text_encoder.tokenizer
    # Ensure the tokenizer has a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        print(f"\nStarting Epoch {epoch+1}/{config.epochs}")

        for batch in train_loader:
            # Unpack the batch: each sample is (image, user_prompt, assistant_answer)
            images, user_prompts, assistant_answers = batch

            # Move images to device (images: [B, 3, 224, 224])
            images = images.to(device)

            # 1. Compute the image embedding and project it.
            with torch.no_grad():
                # Vision encoder outputs a tensor of shape (B, vision_embed_dim)
                image_emb = model.vision_encoder.forward(images)
            # Projection layer outputs a vector per image; we treat this as a prefix.
            projected_emb = model.projection(image_emb)  # shape: (B, text_embed_dim)
            # Unsqueeze to add a sequence dimension (here, prefix_length = 1).
            prefix_embeds = projected_emb.unsqueeze(1)  # shape: (B, 1, text_embed_dim)

            # 2. Build the full text input and corresponding labels for generative training.
            # For each sample, we construct:
            #    full_text = "user: {user_prompt}\nassistant: {assistant_answer}"
            # We then want to compute the loss only on the assistant answer tokens.
            full_texts = []
            labels_list = []
            for prompt, answer in zip(user_prompts, assistant_answers):
                # Build the prompt portion and full text.
                text_prompt = "user: " + prompt.strip() + "\nassistant:"
                full_text = text_prompt + " " + answer.strip()
                full_texts.append(full_text)

                # Tokenize the full text.
                tokenized_full = tokenizer(full_text, return_tensors="pt")
                input_ids = tokenized_full["input_ids"].squeeze(0)  # shape: (seq_len,)

                # Tokenize the prompt portion only to know how many tokens to mask.
                tokenized_prompt = tokenizer(text_prompt, return_tensors="pt")
                prompt_len = tokenized_prompt["input_ids"].shape[1]

                # Create labels: copy input_ids and mask the prompt portion.
                labels = input_ids.clone()
                labels[:prompt_len] = -100  # ignore loss on prompt tokens.
                labels_list.append(labels)

            # 3. Tokenize the full_texts as a batch (pad to the maximum sequence length).
            tokenized_batch = tokenizer(full_texts, return_tensors="pt", padding=True)
            input_ids_batch = tokenized_batch["input_ids"].to(device)  # shape: (B, seq_len)
            attention_mask_batch = tokenized_batch["attention_mask"].to(device)

            # Pad labels_list to form a batch tensor.
            labels_batch = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100).to(device)
            # Ensure labels_batch shape matches input_ids_batch shape.
            if labels_batch.shape[1] < input_ids_batch.shape[1]:
                pad_size = input_ids_batch.shape[1] - labels_batch.shape[1]
                labels_batch = torch.cat([labels_batch, torch.full((labels_batch.size(0), pad_size), -100, device=device)], dim=1)
            elif labels_batch.shape[1] > input_ids_batch.shape[1]:
                labels_batch = labels_batch[:, :input_ids_batch.shape[1]]

            # 4. Instead of passing input_ids, obtain the token embeddings.
            token_embeds = model.text_encoder.model.get_input_embeddings()(input_ids_batch)
            # Concatenate the image prefix embeddings with the token embeddings.
            # prefix_embeds: (B, 1, text_embed_dim), token_embeds: (B, seq_len, text_embed_dim)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            # Since we prepended one token’s worth of embeddings, adjust labels accordingly.
            prefix_pad = torch.full((labels_batch.size(0), 1), -100, dtype=torch.long, device=device)
            labels_batch = torch.cat([prefix_pad, labels_batch], dim=1)

            # 5. Forward pass through the text encoder (generative model) using inputs_embeds.
            outputs = model.text_encoder.model(inputs_embeds=inputs_embeds,
                                                 attention_mask=torch.cat([torch.ones(prefix_embeds.size()[:-1], device=device), attention_mask_batch], dim=1),
                                                 labels=labels_batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            print(f"Batch Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Epoch Loss = {avg_epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "epoch_loss": avg_epoch_loss})

        # 6. Optionally run a brief validation and sample generation.
        model.text_encoder.model.eval()
        with torch.no_grad():
            for i, (val_images, val_prompts, val_assistant_answers) in enumerate(val_loader):
                if i > 0:  # validate only one batch for brevity
                    break
                val_images = val_images.to(device)
                image_emb_val = model.vision_encoder.forward(val_images)
                projected_emb_val = model.projection(image_emb_val)
                prefix_embeds_val = projected_emb_val.unsqueeze(1)

                # For the first sample in the batch, generate output.
                sample_prompt = "user: " + val_prompts[0].strip() + "\nassistant:"
                # Tokenize the sample prompt.
                sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
                input_ids_sample = sample_tokens["input_ids"].to(device)
                token_embeds_sample = model.text_encoder.model.get_input_embeddings()(input_ids_sample)
                inputs_embeds_sample = torch.cat([prefix_embeds_val[0:1], token_embeds_sample], dim=1)

                generated_ids = model.text_encoder.model.generate(inputs_embeds=inputs_embeds_sample,
                                                                   max_length=inputs_embeds_sample.shape[1] + 50,
                                                                   do_sample=True, top_k=50)
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"\nValidation sample generation:\nPrompt: {sample_prompt}\nGenerated: {generated_text}\n")
                wandb.log({"sample_generation": generated_text})
        model.text_encoder.model.train()

    wandb.finish()

if __name__ == "__main__":
    train()
