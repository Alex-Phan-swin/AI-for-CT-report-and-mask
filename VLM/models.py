import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from config import device

# -------------------------
# TOKENIZERS
# -------------------------
llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_tokenizer.pad_token = llm_tokenizer.eos_token

# -------------------------
# MODELS
# -------------------------
text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

for p in text_encoder.parameters():
    p.requires_grad = False

for p in llm.parameters():
    p.requires_grad = False

# -------------------------
# DIMENSIONS
# -------------------------
def get_dims(vision_features):
    return vision_features.shape[1], 768, llm.config.n_embd

# -------------------------
# VLM MODULE
# -------------------------
class VLM(nn.Module):
    def __init__(self, vision_dim, hidden_dim, llm_dim):
        super().__init__()

        self.image_proj = nn.Linear(vision_dim, hidden_dim)
        self.context = nn.Parameter(torch.randn(1, hidden_dim))

        self.to_llm = nn.Linear(hidden_dim, llm_dim)

    def forward(self, vision_feats):
        vision_emb = self.image_proj(vision_feats)

        context = self.context.expand(vision_emb.size(0), -1)

        fused = vision_emb + context  # SIMPLE + STABLE (important fix)

        return self.to_llm(fused)

def build_model(vision_dim, hidden_dim, llm_dim):
    return VLM(vision_dim, hidden_dim, llm_dim).to(device)