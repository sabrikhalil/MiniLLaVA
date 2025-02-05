import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

# Import LoRA functions from PEFT.
from peft import get_peft_model, LoraConfig

# Reuse the LLaVACaptionDataset (which now returns a 4-tuple: (image, user_prompt, assistant_answer, image_path)).
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

def get_top_tokens_from_text(cls_embed, text, tokenizer, llm_model, top_k=5):
    """
    Given a projected image CLS embedding (cls_embed, shape: [1, d]) and a text string,
    tokenize the text and compute cosine similarity between cls_embed and each token's embedding.
    Returns a list of (token, similarity) tuples for the top_k tokens.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return []
    token_ids_tensor = torch.tensor(token_ids).to(cls_embed.device)
    # Get embeddings for these tokens.
    token_embeddings = llm_model.get_input_embeddings()(token_ids_tensor)
    # Normalize both vectors.
    cls_norm = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
    token_norm = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)
    # Compute cosine similarities. Shape: (num_tokens,)
    cosine_sim = torch.matmul(cls_norm, token_norm.T).squeeze(0)
    # Get the top_k indices.
    topk_values, topk_indices = torch.topk(cosine_sim, min(top_k, cosine_sim.size(0)))
    top_tokens = [tokenizer.decode([token_ids[idx]]) for idx in topk_indices.tolist()]
    top_similarities = topk_values.tolist()
    return list(zip(top_tokens, top_similarities))

def validate(model, val_loader, tokenizer, epoch, iteration, device, max_length):
    """
    For validation, the full text prompt is used (i.e. only the user query with "assistant:" appended)
    and the model generates a continuation. Additionally, for each example in the batch the projected
    image embedding (using the CLS token) is compared to tokens from both the user prompt and the 
    ground-truth assistant answer. The top-5 tokens (with cosine similarities) and the average
    similarity are logged for inspection, along with the image filename.
    """
    model.llm.model.eval()
    print(f"\n[Validation @ epoch {epoch}, iteration {iteration}]")
    avg_prompt_sim_total = 0.0
    avg_gt_sim_total = 0.0
    count = 0

    for batch in val_loader:
        # Each batch returns (image, user_prompt, assistant_answer, image_path).
        if len(batch) >= 4:
            images, prompts, answers, image_paths = batch[:4]
        else:
            continue

        with torch.no_grad():
            # Process up to four examples.
            for i in range(min(4, len(images))):
                sample_image = images[i].unsqueeze(0)  # [1, C, H, W]
                proj_embed = model.projection(model.vision_encoder.forward(sample_image))
                # Use the CLS token (first token) as the representative embedding.
                cls_embed = proj_embed[:, 0, :]  # shape: [1, d]
                
                # Retrieve top-5 tokens from both the user prompt and the ground-truth assistant answer.
                prompt_top = get_top_tokens_from_text(cls_embed, prompts[i], tokenizer, model.llm.model, top_k=5)
                gt_top = get_top_tokens_from_text(cls_embed, answers[i], tokenizer, model.llm.model, top_k=5)
                
                # Compute average cosine similarity for the top-5 tokens.
                avg_prompt_sim = sum(sim for _, sim in prompt_top) / len(prompt_top) if prompt_top else 0.0
                avg_gt_sim = sum(sim for _, sim in gt_top) / len(gt_top) if gt_top else 0.0

                avg_prompt_sim_total += avg_prompt_sim
                avg_gt_sim_total += avg_gt_sim
                count += 1

                image_name = os.path.basename(image_paths[i])
                print(f"\nImage: {image_name}")
                print("User Prompt:", prompts[i])
                print("Assistant Answer (Ground Truth):", answers[i])
                print("User Prompt Top Tokens (token, cosine similarity):", prompt_top)
                print("Assistant Answer Top Tokens (token, cosine similarity):", gt_top)
                print(f"Avg Prompt Similarity: {avg_prompt_sim:.4f}")
                print(f"Avg Assistant Answer Similarity: {avg_gt_sim:.4f}")

                wandb.log({
                    f"val_image_name_{i}": image_name,
                    f"val_prompt_{i}": prompts[i],
                    f"val_ground_truth_{i}": answers[i],
                    f"prompt_top_{i}": prompt_top,
                    f"assistant_top_{i}": gt_top,
                    f"avg_prompt_similarity_{i}": avg_prompt_sim,
                    f"avg_assistant_similarity_{i}": avg_gt_sim,
                    "epoch": epoch,
                    "iteration": iteration,
                })
            break  # Process only one batch for diagnostic purposes.

    if count > 0:
        overall_avg_prompt = avg_prompt_sim_total / count
        overall_avg_gt = avg_gt_sim_total / count
        wandb.log({
            "val_overall_avg_prompt_similarity": overall_avg_prompt,
            "val_overall_avg_assistant_similarity": overall_avg_gt,
        })
        print(f"\nValidation Overall Avg Prompt Similarity (Top-5): {overall_avg_prompt:.4f}")
        print(f"Validation Overall Avg Assistant Answer Similarity (Top-5): {overall_avg_gt:.4f}")

    model.llm.model.train()

#############################################
# Stage 1: Fine-Tuning the Image Projector Only
#############################################

def train_projector():
    max_length = 128

    wandb.init(
        project="MiniLLaVA",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 100,
            "max_length": max_length,
            "description": "Stage 1: Fine-tuning only the image projector for feature alignment. Vision encoder and LLM are frozen.",
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Initialize the integrated model.
    # Vision encoder and LLM remain frozen.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    # Freeze the vision encoder.
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Freeze the LLM weights so that only the projection layer is updated.
    for param in model.llm.model.parameters():
        param.requires_grad = False

    # Use an optimizer that only updates the projection layer.
    optimizer = optim.AdamW(model.projection.parameters(), lr=config.learning_rate)

    # Get training and validation splits.
    json_file = "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json"
    img_dir = "src/data/LLaVA_Instruct_Files/images"  # Visuals are used.
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
            # Each batch returns (image, user_prompt, assistant_answer, image_path).
            images, prompts, answers, _ = batch[:4]
            # Compute visual features (via the frozen vision encoder) and project them.
            prefix_embeds = model.projection(model.vision_encoder.forward(images))
            # Construct the full text for next-token prediction.
            texts = [f"user: {p.strip()}\nassistant: {a.strip()}" for p, a in zip(prompts, answers)]
            outputs = model.llm(texts, prefix_embeds=prefix_embeds, max_length=max_length)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            print(f"Iteration {iteration}, Batch Loss: {loss.item():.4f}")

            # Every 100 iterations, save a checkpoint and run validation.
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
        
        # Save a checkpoint at the end of the epoch.
        epoch_save_path = os.path.join("saved_models", f"projector_epoch{epoch+1}.pth")
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.projection.state_dict(), epoch_save_path)
        print(f"Saved projection weights at end of epoch {epoch+1} to {epoch_save_path}")
        torch.cuda.empty_cache()

        # Run validation at the end of the epoch.
        validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
        model.llm.model.train()  # Ensure LLM is in train mode.

    wandb.finish()

if __name__ == "__main__":
    train_projector()
