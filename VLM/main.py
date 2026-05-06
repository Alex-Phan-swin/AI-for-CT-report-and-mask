import torch
from config import device, load_data
from models import (
    llm, llm_tokenizer,
    get_dims, build_projections, fuse
)

vision_features, labels = load_data()

# =========================
# SETUP
# =========================
hidden_dim = 512
vision_dim, text_dim, llm_dim = get_dims(vision_features)

proj = build_projections(vision_dim, text_dim, hidden_dim, llm_dim)

# =========================
# PROMPT
# =========================
def build_prompt():
    return "Clinical Impression: Liver CT scan shows "

# =========================
# GENERATION
# =========================
def generate_text(vision_feats, max_len=80):

    vision_emb = proj["image_proj"](vision_feats)

    context = proj["context"].expand(vision_emb.shape[0], -1)

    fused = fuse(vision_emb, context, proj["fusion_proj"])

    vision_token = proj["llm_proj"](fused).unsqueeze(1)

    prompt = build_prompt()
    input_ids = llm_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    prompt_embeds = llm.transformer.wte(input_ids)

    generated = torch.cat([vision_token, prompt_embeds], dim=1)

    output_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            outputs = llm(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]

            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)

            token_id = next_token.item()
            output_tokens.append(token_id)

            if token_id == llm_tokenizer.eos_token_id:
                break

            next_emb = llm.transformer.wte(next_token)
            generated = torch.cat([generated, next_emb], dim=1)

    return llm_tokenizer.decode(output_tokens, skip_special_tokens=True)

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