import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/phi-1_5", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Loads a generative language model and its tokenizer.
        This model is used in generative mode, so it returns logits for the full sequence.
        """
        super(TextEncoder, self).__init__()
        self.device = device
        # Load tokenizer and generative model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                #load_in_8bit=True,  # this loads the model in 8-bit mod
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode unless fine-tuning

    def forward(self, text: str, prefix_embeds: torch.Tensor = None):
        """
        Processes text input and (optionally) prepends image-based prefix embeddings.
        
        Args:
            text (str): Input text string (e.g., the user prompt).
            prefix_embeds (torch.Tensor, optional): Precomputed image prefix embeddings 
                of shape (batch_size, prefix_length, embed_dim). Defaults to None.
                
        Returns:
            logits (torch.Tensor): Logits for the generated sequence.
        """
        # Prepend the "user:" prompt to mimic the LLaVA prompt format.
        input_text = "user: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if prefix_embeds is not None:
            # Retrieve the text token embeddings from the model's embedding layer.
            token_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            # Ensure prefix_embeds is on the same device.
            prefix_embeds = prefix_embeds.to(self.device)
            # Concatenate prefix embeddings and token embeddings along the sequence dimension.
            # Assume prefix_embeds shape: (batch_size, prefix_length, embed_dim)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            # Pass the concatenated embeddings to the model.
            outputs = self.model(inputs_embeds=inputs_embeds)
        else:
            # No prefix provided; process normally.
            outputs = self.model(**inputs)
        
        # Return the logits (shape: [batch_size, sequence_length, vocab_size]).
        return outputs.logits

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
        encoder = TextEncoder()
        # For testing, we call forward without any prefix.
        logits = encoder.forward(text_input)
        print("Logits Shape:", logits.shape)
    else:
        print("Please provide text as a command-line argument.")
