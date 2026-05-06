import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from config import device

# -------------------------
# TOKENIZER (ADD THIS)
# -------------------------
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# -------------------------
# TEXT ENCODER (frozen)
# -------------------------
text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
text_encoder.eval()

for p in text_encoder.parameters():
    p.requires_grad = False


# -------------------------
# CLIP STYLE MODEL
# -------------------------
class CLIPVLM(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim=512):
        super().__init__()

        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, vision_feats, text_embeds):
        v = self.vision_proj(vision_feats)
        t = self.text_proj(text_embeds)

        v = v / v.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)

        return v, t        