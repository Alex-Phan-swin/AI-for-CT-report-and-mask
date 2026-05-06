import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from config import load_data, device
from models import CLIPVLM, text_encoder, text_tokenizer
from clip_loss import clip_loss

# -------------------------
# DATA
# -------------------------
vision_features, labels = load_data()

dataset = TensorDataset(vision_features, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------------------------
# MODEL
# -------------------------
vision_dim = vision_features.shape[1]
text_dim = text_encoder.config.hidden_size

model = CLIPVLM(vision_dim, text_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# TRAIN LOOP
# -------------------------
epochs = 1
total_steps = epochs * len(loader)
step = 0

pbar = tqdm(total=total_steps)

for epoch in range(epochs):
    for v, t in loader:

        v = v.to(device)

        # -------------------------
        # FIX 1: ensure text is string list
        # -------------------------
        if torch.is_tensor(t):
            t = [str(x.item()) for x in t]

        # -------------------------
        # TOKENIZE TEXT (FIXED)
        # -------------------------
        text_inputs = text_tokenizer(
            t,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # move tensors to device properly
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # -------------------------
        # TEXT ENCODING
        # -------------------------
        with torch.no_grad():
            text_emb = text_encoder(**text_inputs).last_hidden_state[:, 0, :]

        # -------------------------
        # FORWARD
        # -------------------------
        v_emb, t_emb = model(v, text_emb)

        loss = clip_loss(v_emb, t_emb, model.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------
        # PROGRESS TRACKING
        # -------------------------
        step += 1
        steps_left = total_steps - step

        pbar.set_description(
            f"Loss {loss.item():.4f} | Steps left {steps_left}"
        )
        pbar.update(1)

pbar.close()

# -------------------------
# SAVE MODEL
# -------------------------
save_dir = os.path.join(os.path.dirname(__file__), "projection")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "clip_vlm.pt")

torch.save(model.state_dict(), save_path)

print("CLIP training complete")