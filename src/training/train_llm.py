mport os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

# Import LoRA functions from PEFT.
from peft import get_peft_model, LoraConfig

# Reuse the LLaVACaptionDataset (returns (image, user_prompt, assistant_answer)).
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
    For validation, use the full text prompt (i.e. only the user query with "assistant:" appended)
    and let the model generate a continuation.
    """
    model.llm.model.eval()
    print(f"\n[Validation @ epoch {epoch}, iteration {iteration}]")
    for batch in val_loader:
        if len(batch) >= 3:
            # Batch: (image, user_prompt, assistant_answer, *optional)
            _, prompts, answers = batch[:3]
        else:
            continue
        for i in range(min(5, len(prompts))):
            input_text = f"user: {prompts[i].strip()}\nassistant:"  # Only the prompt is fed.
            generated_ids = model.llm.generate(
                [input_text],
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\n=== Validation Sample ===")
            print("Input Text:", input_text)
            print("Ground Truth Assistant Answer:", answers[i].strip())
            print("Generated Assistant Answer:", generated_text)
            wandb.log({
                "val_prompt": input_text,
                "val_ground_truth": answers[i].strip(),
                "val_generated": generated_text,
                "epoch": epoch,
                "iteration": iteration,
            })
        break  # Process only one batch
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
            "learning_rate": 1e-4,
            "batch_size": 4,  # Use batch size 1 for detailed debugging.
            "epochs": 3,
            "max_length": max_length,
            "description": "LM finetuning on Q&A (ignoring visuals) using LoRA for standard next-token prediction.",
        }
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Initialize the integrated model.
    # For LM-only finetuning, visuals are ignored.
    model = MiniLLaVA(device=device)
    model.to(device)
    model.train()

    # Freeze the vision encoder and projection layer.
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.projection.parameters():
        param.requires_grad = False

    # Inject LoRA adapters into the LM's query, key, and value projections.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.llm.model = get_peft_model(model.llm.model, lora_config)
    model.llm.model.print_trainable_parameters()

    # Fine-tune only the LM (i.e. the LoRA parameters).
    optimizer = optim.AdamW(model.llm.model.parameters(), lr=config.learning_rate)

    # Get training and validation splits.
    json_file = "src/data/LLaVA_Instruct_Files/llava_instruct_150k.json"
    img_dir = "src/data/LLaVA_Instruct_Files/images"  # Ignored for LM-only finetuning.
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
            # Each batch returns (image, user_prompt, assistant_answer); visuals are ignored here.
            _, prompts, answers = batch[:3]
            # Construct the full text for next-token prediction.
            # Format: "user: <prompt>\nassistant: <answer>"
            texts = [f"user: {p.strip()}\nassistant: {a.strip()}" for p, a in zip(prompts, answers)]
            # Forward pass using the new LLM forward method.
            outputs = model.llm(texts, prefix_embeds=None, max_length=max_length)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            print(f"Iteration {iteration}, Batch Loss: {loss.item():.4f}")

            if iteration % 100 == 0:
                validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
            iteration += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Epoch Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "epoch_loss": avg_epoch_loss})
        save_path = os.path.join("saved_models", f"llm_epoch{epoch+1}.pth")
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.llm.state_dict(), save_path)
        print(f"Saved LLM weights to {save_path}")
        torch.cuda.empty_cache()
        validate(model, val_loader, tokenizer, epoch, iteration, device, max_length)
        model.llm.model.train()

    wandb.finish()

if __name__ == "__main__":
    train_llm()