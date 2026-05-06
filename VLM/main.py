import torch
from config import load_data, device
from models import CLIPVLM, text_encoder, text_tokenizer
import os

# =========================
# DATA
# =========================
vision_features, labels = load_data()

vision_dim = vision_features.shape[1]
text_dim = text_encoder.config.hidden_size

# Define text bank for reports
text_bank = [
    "Healthy liver: Normal appearance with no signs of disease.",
    "Hepatic Steatosis: Fatty infiltration of the liver detected, indicating steatosis."
]

# =========================
# MODEL PATH
# =========================
model_path = os.path.join(
    os.path.dirname(__file__),
    "projection",
    "clip_vlm.pt"
)

# =========================
# MODEL LOAD (FIXED)
# =========================
model = CLIPVLM(vision_dim, text_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# RETRIEVAL FUNCTION
# =========================
def retrieve_best_match(image_feat, text_bank):

    image_feat = image_feat.to(device)

    with torch.no_grad():

        # image embedding
        v = model.vision_proj(image_feat)
        v = v / v.norm(dim=-1, keepdim=True)

        best_score = -float("inf")
        best_text = None

        for t in text_bank:

            # SUPPORT BOTH STRING OR TOKENIZED INPUT
            if isinstance(t, str):
                inputs = text_tokenizer(
                    t,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in t.items()}

            outputs = text_encoder(**inputs)
            t_emb = outputs.last_hidden_state[:, 0, :]
            t_emb = model.text_proj(t_emb)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

            score = torch.matmul(v, t_emb.T).item()

            if score > best_score:
                best_score = score
                best_text = t

    return best_text

# =========================
# RUN
# =========================
if __name__ == "__main__":

    for i in range(10):
        print(f"\n--- MATCHED REPORT {i}---")
        print(retrieve_best_match(vision_features[i:i+1], text_bank))