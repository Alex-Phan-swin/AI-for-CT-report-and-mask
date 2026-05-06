import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load features
def load_data():
    vision_features = torch.load('Encoder/features.pt').to(device)
    labels = torch.load('Encoder/labels.pt')
    return vision_features, labels