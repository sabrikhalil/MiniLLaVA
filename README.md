## MiniLLaVA

MiniLLaVA is a minimal Vision-Language Model prototype inspired by LLaVA. It integrates a CLIP-based vision encoder and a φ‑1.5 language model by projecting vision embeddings into the language space before feeding them to the language model. This repository serves as a starting point for experiments in vision-language alignment.

## Architecture Implemented

**Vision Encoder:** Uses [CLIP](https://github.com/openai/CLIP) to extract image features.

**Language Model:** Incorporates the [φ‑1.5](https://huggingface.co/microsoft/phi-1_5) model for text generation.

**Projection Module:** Projects CLIP image embeddings into the φ‑1.5 language model’s token space, allowing the image features to condition text generation.

**Modular Structure:** Easily extendable for further experimentation (e.g., fine-tuning on custom vision-language datasets).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/MiniLLaVA.git
   cd MiniLLaVA

2. **Install dependecies:**
   ```bash
   pip install -r requirements.txt

3. **Download Dataset:**
   ```bash
   python src/data/download_dataset.py
4. **Start training:**
   ```bash
     python src/training/train_projector.py  -- Train first the projector, freeze both LLM and Vision Encoder.
     python src/training/train.py            -- Train after both projector and LLM (using Lora tuning).
5. **Inference:**
  ```bash
     python src/evaluation/manual_eval.py    -- Interactive session in terminal to enter prompt and receive responses.
     python src/evaluation/ui_chat.py        -- UI displays image at left and chat on right. 
