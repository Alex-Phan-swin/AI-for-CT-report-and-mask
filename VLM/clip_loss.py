import torch

def clip_loss(image_embeds, text_embeds, temperature):
    logits = image_embeds @ text_embeds.T / temperature

    labels = torch.arange(logits.size(0)).to(logits.device)

    loss_i = torch.nn.functional.cross_entropy(logits, labels)
    loss_t = torch.nn.functional.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2