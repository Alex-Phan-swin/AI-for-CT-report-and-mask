import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from config import load_data, device
from models import build_model, get_dims, llm, llm_tokenizer

# =========================
# LOAD DATA
# =========================
vision_features, labels = load_data()

vision_dim, _, llm_dim = get_dims(vision_features)
model = build_model(vision_dim, 512, llm_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

dataset = TensorDataset(vision_features)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# =========================
# TEXT TEMPLATES
# =========================
TEMPLATES = [
    "Liver appears normal.",
    "Mild hepatic steatosis noted.",
    "No significant findings.",
    "Unremarkable CT scan.",
    "Severe hepatic steatosis observed.",
]

# =========================
# TRAIN STEP
# =========================
def train_step(v, text_batch):

    v = v.to(device)

    prefix = model(v).unsqueeze(1)

    tokens = llm_tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    embeds = llm.transformer.wte(tokens)

    inputs = torch.cat([prefix, embeds[:, :-1, :]], dim=1)
    targets = tokens[:, 1:]

    outputs = llm(inputs_embeds=inputs)
    logits = outputs.logits[:, :-1, :]

    loss = loss_fn(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# =========================
# TRAIN LOOP (STEP TRACKING ADDED)
# =========================
if __name__ == "__main__":

    print("\n🚀 TRAINING STARTED\n")

    epochs = 3
    total_steps = epochs * len(loader)
    step = 0

    pbar = tqdm(total=total_steps)

    for epoch in range(epochs):

        for i, (v,) in enumerate(loader):

            text_batch = [
                TEMPLATES[i % len(TEMPLATES)]
                for _ in range(v.size(0))
            ]

            loss = train_step(v, text_batch)

            step += 1
            steps_left = total_steps - step

            pbar.update(1)
            pbar.set_description(
                f"Epoch {epoch+1} | Loss {loss:.4f} | Steps left {steps_left}"
            )

    pbar.close()

    # =========================
    # SAVE MODEL TO VLM/PROJECTION
    # =========================
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "projection"
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "vlm_model.pt")

    torch.save(model.state_dict(), save_path)

    print(f"\n✅ Model saved to: {save_path}")