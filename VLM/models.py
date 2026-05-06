import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from config import device

# =========================
# TOKENIZERS
# =========================
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")

llm_tokenizer.pad_token = llm_tokenizer.eos_token

# =========================
# MODELS
# =========================
text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Freeze models
for p in text_encoder.parameters():
    p.requires_grad = False

for p in llm.parameters():
    p.requires_grad = False

# =========================
# DIMENSIONS
# =========================
def get_dims(vision_features):
    vision_dim = vision_features.shape[1]
    text_dim = text_encoder.config.hidden_size
    llm_dim = llm.config.n_embd
    return vision_dim, text_dim, llm_dim

# =========================
# BUILD PROJECTIONS
# =========================
def build_projections(vision_dim, text_dim, hidden_dim, llm_dim):
    return {
        "image_proj": nn.Linear(vision_dim, hidden_dim).to(device),
        "text_proj": nn.Linear(text_dim, hidden_dim).to(device),
        "fusion_proj": nn.Linear(hidden_dim * 2, hidden_dim).to(device),
        "llm_proj": nn.Linear(hidden_dim, llm_dim).to(device),
        "context": nn.Parameter(torch.randn(1, hidden_dim)).to(device)
    }

# =========================
# TEXT ENCODING
# =========================
def encode_text(text_inputs):
    inputs = text_tokenizer(
        text_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = text_encoder(**inputs)

    return outputs.last_hidden_state[:, 0, :]

# =========================
# FUSION
# =========================
def fuse(vision_emb, text_emb, fusion_layer):
    fused = torch.cat([vision_emb, text_emb], dim=-1)
    return fusion_layer(fused)