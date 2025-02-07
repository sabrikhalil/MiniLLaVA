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
        # It is assumed that special tokens "[IMG_START]" and "[IMG_END]" have been added
        # (e.g. via MiniLLaVA __init__) and the model resized accordingly.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        print(f"[LLM.__init__] Model loaded on {self.device} with config: {self.model.config}")

    def _wrap_prefix_with_img_tokens(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        """
        Given a prefix_embeds tensor (batch_size, prefix_length, embed_dim),
        retrieve the embeddings for the special tokens "[IMG_START]" and "[IMG_END]"
        and wrap the prefix with them.
        """
        img_start_id = self.tokenizer.convert_tokens_to_ids("[IMG_START]")
        img_end_id = self.tokenizer.convert_tokens_to_ids("[IMG_END]")
        start_emb = self.model.get_input_embeddings()(torch.tensor([img_start_id], device=self.device))
        end_emb = self.model.get_input_embeddings()(torch.tensor([img_end_id], device=self.device))
        batch_size = prefix_embeds.size(0)
        start_emb = start_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        end_emb = end_emb.unsqueeze(0).expand(batch_size, -1, -1)      # (B, 1, embed_dim)
        wrapped_prefix = torch.cat([start_emb, prefix_embeds, end_emb], dim=1)
        return wrapped_prefix

    def forward(self, prompt_texts, assistant_texts, prefix_embeds: torch.Tensor = None, max_length: int = None):
        """
        Processes a batch of prompt and assistant text inputs and (optionally) prepends image-based prefix embeddings.
        The loss is computed only on the assistant part.
        """
        batch_full_ids = []
        batch_labels = []
        for prompt, assistant in zip(prompt_texts, assistant_texts):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            assistant_ids = self.tokenizer.encode(assistant, add_special_tokens=False)
            full_ids = prompt_ids + assistant_ids
            # Mask out (ignore) tokens corresponding to the prompt.
            labels = [-100] * len(prompt_ids) + assistant_ids
            batch_full_ids.append(torch.tensor(full_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
        
        from torch.nn.utils.rnn import pad_sequence
        padded_input_ids = pad_sequence(batch_full_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()
        
        if max_length is not None and padded_input_ids.size(1) > max_length:
            padded_input_ids = padded_input_ids[:, :max_length]
            padded_labels = padded_labels[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        
        token_embeds = self.model.get_input_embeddings()(padded_input_ids.to(self.device))
        
        if prefix_embeds is not None:
            wrapped_prefix = self._wrap_prefix_with_img_tokens(prefix_embeds).to(self.device)
            # Concatenate the wrapped prefix and token embeddings.
            inputs_embeds = torch.cat([wrapped_prefix, token_embeds], dim=1)
            batch_size = padded_input_ids.size(0)
            prefix_length = wrapped_prefix.size(1)
            prefix_mask = torch.ones(batch_size, prefix_length, device=self.device)
            attention_mask = torch.cat([prefix_mask, attention_mask.to(self.device)], dim=1)
            # For the prefix positions, set labels to -100 so they do not contribute to loss.
            prefix_labels = torch.full((batch_size, prefix_length), -100, device=self.device, dtype=torch.long)
            labels = torch.cat([prefix_labels, padded_labels.to(self.device)], dim=1)
        else:
            inputs_embeds = token_embeds
            labels = padded_labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, texts, prefix_embeds=None, max_length=128, **generate_kwargs):
        """
        Generates text continuations for a list of text prompts.
        (This function remains unchanged.)
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
            prefix_embeds = self._wrap_prefix_with_img_tokens(prefix_embeds)
            token_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            prefix_mask = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), device=self.device)
            attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)
            out = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                **generate_kwargs
            )
            return out
        else:
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                **generate_kwargs
            )
            return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
        llm = LLM()
        # For a simple test, provide the full text as prompt and an empty assistant part.
        outputs = llm.forward([text_input], [""], prefix_embeds=None)
        print("Logits Shape:", outputs.logits.shape)
    else:
        print("Please provide text as a command-line argument.")
