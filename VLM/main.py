import torch
import os

from config import load_data, device
from models import build_model, get_dims, llm, llm_tokenizer

vision_features, _ = load_data()

vision_dim, _, llm_dim = get_dims(vision_features)

model = build_model(vision_dim, 512, llm_dim)

# =========================
# LOAD FROM projection FOLDER
# =========================
model_path = os.path.join(
    os.path.dirname(__file__),
    "projection",
    "vlm_model.pt"
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# =========================
# GENERATION
# =========================
def generate(v):

    v = v.to(device)

    prefix = model(v).unsqueeze(1)

    prompt = "Clinical Impression: "
    input_ids = llm_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    prompt_emb = llm.transformer.wte(input_ids)

    x = torch.cat([prefix, prompt_emb], dim=1)

    out = []

    for _ in range(60):
        with torch.no_grad():
            logits = llm(inputs_embeds=x).logits[:, -1, :]

            probs = torch.softmax(logits / 1.0, dim=-1)
            next_token = torch.multinomial(probs, 1)

            out.append(next_token.item())

            x = torch.cat([x, llm.transformer.wte(next_token)], dim=1)

    return llm_tokenizer.decode(out, skip_special_tokens=True)

# =========================
# RUN
# =========================
if __name__ == "__main__":

    print("\n=== VLM INFERENCE ===\n")

    for i in range(3):
        print("\n--- REPORT ---")
        print(generate(vision_features[i:i+1]))