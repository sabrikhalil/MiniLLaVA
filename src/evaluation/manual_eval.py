#!/usr/bin/env python3
import os
import sys
import torch
from src.models.vlm import MiniLLaVA

def main():
    # Adjust the relative path assuming this script is in "src/evaluation" and your data is at the project root.
    image_path = "././data/statue.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the integrated model.
    print("[INFO] Loading MiniLLaVA model...")
    model = MiniLLaVA(device=device)
    model.to(device)
    
    # IMPORTANT: Inject the LoRA adapters exactly as in training.
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.llm.model = get_peft_model(model.llm.model, lora_config)

    # Load the trained checkpoint.
    checkpoint_path = "././saved_models/checkpoint_epoch10.pth"
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        # Load LLM weights
        if "llm_state_dict" in checkpoint:
            model.llm.load_state_dict(checkpoint["llm_state_dict"])
        else:
            print("[WARNING] 'llm_state_dict' not found in the checkpoint.")
        
        # Load projector weights
        if "projector_state_dict" in checkpoint:
            model.projection.load_state_dict(checkpoint["projector_state_dict"])
        else:
            print("[WARNING] 'projector_state_dict' not found in the checkpoint.")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
    
    pretrained_projector_path = "././saved_models/projector_iter_102300_epoch31.pth"
    if os.path.exists(pretrained_projector_path):
        print(f"[INFO] Loading pretrained projector from {pretrained_projector_path}")
        model.projection.load_state_dict(torch.load(pretrained_projector_path, map_location=device))
    else:
        print("[WARNING] Pretrained projector not found. Using default projector weights.")

    # Ensure the tokenizer has a pad token.
    tokenizer = model.llm.tokenizer
    if tokenizer.pad_token is None:
        print("[INFO] Setting tokenizer.pad_token to tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Process the image to get its visual prefix.
    print(f"[INFO] Processing image: {image_path}")
    with torch.no_grad():
        image_emb = model.vision_encoder.forward(image_path)
        prefix_embeds = model.projection(image_emb)
        

    print("\n[INFO] Image loaded and processed. You can now chat with the image.")
    print("Type your prompt and hit ENTER. Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            prompt = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        chat_prompt = f"user: {prompt}\nassistant:"

        with torch.no_grad():
            generated_ids = model.llm.generate(
                [chat_prompt],
                prefix_embeds=prefix_embeds,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(chat_prompt):
            generated_text = generated_text[len(chat_prompt):].strip()

        print("Assistant:", generated_text)
        print("-" * 80)

if __name__ == "__main__":
    main()
