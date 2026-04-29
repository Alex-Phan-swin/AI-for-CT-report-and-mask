import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

#Loading feature vectors
vision_features = torch.load('Encoder/vision_features.pt')
labels = torch.load('Encoder/labels.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vision_features = vision_features.to(device)

#Use pretrained model for text tokenizer and encoder from SOTA medical
#Text tokenizer and Encoder
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract").to(device)

#Freeze text encoder parameters
for param in text_encoder.parameters():
    param.requires_grad = False

