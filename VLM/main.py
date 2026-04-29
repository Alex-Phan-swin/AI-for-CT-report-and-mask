import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# =========================
# DEVICE
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# LOAD FEATURES
# =========================
vision_features = torch.load('../Encoder/features.pt').to(device)
labels = torch.load('../Encoder/labels.pt')

# =========================
# TOKENIZERS
# =========================
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")

llm_tokenizer.pad_token = llm_tokenizer.eos_token

# =========================
# MODELS (LIGHTWEIGHT)
# =========================
print("Loading text encoder...")
text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)

print("Loading LLM...")
llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

print("Models loaded.")

# Freeze models
for p in text_encoder.parameters():
    p.requires_grad = False

for p in llm.parameters():
    p.requires_grad = False

# =========================
# DIMENSIONS
# =========================
vision_dim = vision_features.shape[1]
text_dim = text_encoder.config.hidden_size
hidden_dim = 512
llm_dim = llm.config.n_embd

# =========================
# PROJECTION LAYERS
# =========================
image_projection = nn.Linear(vision_dim, hidden_dim).to(device)
text_projection = nn.Linear(text_dim, hidden_dim).to(device)
fusion_projection = nn.Linear(hidden_dim * 2, hidden_dim).to(device)
llm_projection = nn.Linear(hidden_dim, llm_dim).to(device)

# =========================
# CONTEXT TOKEN
# =========================
clinical_context = nn.Parameter(torch.randn(1, hidden_dim)).to(device)

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

    return outputs.last_hidden_state[:, 0, :]  # CLS token

# =========================
# FUSION
# =========================
def fuse(vision_emb, text_emb):
    fused = torch.cat([vision_emb, text_emb], dim=-1)
    return fusion_projection(fused)

# =========================
# PROMPT
# =========================
def build_prompt():
    return "Clinical Impression: Liver CT scan shows "

# =========================
# GENERATION (IMPROVED)
# =========================
def generate_text(vision_feats, max_len=80):

    # ---- Vision ----
    vision_emb = image_projection(vision_feats)

    # ---- Context ----
    context = clinical_context.expand(vision_emb.shape[0], -1)

    # ---- Fuse ----
    fused = fuse(vision_emb, context)

    # ---- Project to LLM space ----
    vision_token = llm_projection(fused).unsqueeze(1)

    # ---- Prompt ----
    prompt = build_prompt()
    input_ids = llm_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    prompt_embeds = llm.transformer.wte(input_ids)

    # Combine vision + text
    generated = torch.cat([vision_token, prompt_embeds], dim=1)

    output_tokens = []

    for step in range(max_len):
        with torch.no_grad():
            outputs = llm(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]

            # 🔥 sampling instead of greedy (better text)
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            output_tokens.append(token_id)

            if token_id == llm_tokenizer.eos_token_id:
                break

            next_emb = llm.transformer.wte(next_token)
            generated = torch.cat([generated, next_emb], dim=1)

    if output_tokens:
        report = llm_tokenizer.decode(output_tokens, skip_special_tokens=True)
    else:
        report = "[No report generated]"

    return report

# =========================
# RUN
# =========================
if __name__ == "__main__":

    print("\n" + "="*70)
    print("LIGHTWEIGHT VLM MEDICAL REPORT GENERATOR")
    print("="*70)

    num_reports = min(3, len(vision_features))

    for idx in range(num_reports):
        print(f"\n[Report {idx+1}] Processing...")

        try:
            report = generate_text(
                vision_features[idx:idx+1],
                max_len=100
            )

            print("\n" + "─"*70)
            print(f"CLINICAL IMPRESSION - Image {idx+1}")
            print("─"*70)
            print(report)
            print("─"*70)

        except Exception as e:
            print(f"Error: {e}")

    print("\nDone.")