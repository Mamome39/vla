import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ID = "Qwen/Qwen3.5-2B"


class Qwen35_2B(nn.Module):
    """Qwen3.5-2B. Weights download to the HF cache on first use."""

    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(REPO_ID)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=device)
        if freeze:
            self.model.requires_grad_(False).eval()
        self.hidden_size = self.model.config.text_config.hidden_size  # 2048

    def _inputs(self, images, texts, add_generation_prompt=True):
        """images: one list of PIL images per sample; texts: one instruction per sample."""
        conversations = [
            [{"role": "user", "content": [*({"type": "image", "image": im} for im in ims), {"type": "text", "text": t}]}]
            for ims, t in zip(images, texts, strict=True)
        ]
        return self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(self.model.device)

    def forward(self, images, texts):
        """Returns last-layer token features (B, T, 2048) and attention mask (B, T)."""
        inputs = self._inputs(images, texts)
        out = self.model.model(**inputs, use_cache=False)  # base model, skips the LM head
        return out.last_hidden_state, inputs["attention_mask"]

    @torch.no_grad()
    def chat(self, images, texts, max_new_tokens=128):
        inputs = self._inputs(images, texts)
        ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
