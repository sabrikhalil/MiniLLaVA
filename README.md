## MiniLLaVA

MiniLLaVA is a minimal Vision-Language Model prototype inspired by LLaVA. It integrates a vision encoder and a language model by projecting vision embeddings into the language space before feeding them to the language model. This repository serves as a starting point for experiments in vision-language alignment.

## Features

- **Vision Encoder:** Uses a lightweight model (e.g., a SigLIP-based encoder) to extract image features.
- **Language Encoder:** Incorporates a small language model (e.g., DistilBERT) to process text.
- **Projection Module:** Projects vision embeddings into the language model’s token space.
- **Modular Structure:** Easily extendable for further experimentation (e.g., fine-tuning on small datasets).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/MiniLLaVA.git
   cd MiniLLaVA
