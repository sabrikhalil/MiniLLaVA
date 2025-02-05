import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM(nn.Module):
    def __init__(self, model_name: str = "microsoft/phi-1_5", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Loads a generative language model and its tokenizer.
        Returns logits for the full sequence.
        """
        super(LLM, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        # Note: Mode (train/eval) will be set externally as needed.
    
    def forward(self, texts, prefix_embeds: torch.Tensor = None, max_length: int = 128):
        """
        Processes a batch of text inputs and (optionally) prepends image-based prefix embeddings.
        
        Args:
            texts (list[str]): List of text strings (each should be already formatted,
                                 e.g. "user: <prompt>\nassistant: <answer>").
            prefix_embeds (torch.Tensor, optional): Precomputed image prefix embeddings
                of shape (batch_size, prefix_length, embed_dim). If provided, this branch is used.
            max_length (int): Maximum sequence length used for tokenization.
                
        Returns:
            outputs: The model outputs including logits and loss (if labels are provided).
        """
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attn_mask = tokenized["attention_mask"].to(self.device)
        
        if prefix_embeds is not None:
            # Retrieve token embeddings.
            token_embeds = self.model.get_input_embeddings()(input_ids)
            prefix_embeds = prefix_embeds.to(self.device)
            # Concatenate prefix embeddings on the left.
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            # Create a mask for the prefix (all ones).
            prefix_mask = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), device=self.device)
            attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)
            # Instead of masking out the visual prefix from the loss,
            # we set their target to the pad token ID so gradients can flow.
            batch_size = input_ids.size(0)
            prefix_length = prefix_embeds.size(1)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            labels = torch.cat([torch.full((batch_size, prefix_length), pad_id, device=self.device), input_ids], dim=1)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels
            )
        else:
            # Standard case: no visual prefix.
            labels = input_ids.clone()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels
            )
        return outputs

    def generate(self, texts, max_length=128, **generate_kwargs):
        """
        Generates text continuations for a list of text prompts.
        """
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attn_mask = tokenized["attention_mask"].to(self.device)
        return self.model.generate(input_ids=input_ids, attention_mask=attn_mask, **generate_kwargs)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
        llm = LLM()
        outputs = llm.forward([text_input])
        print("Logits Shape:", outputs.logits.shape)
    else:
        print("Please provide text as a command-line argument.")
