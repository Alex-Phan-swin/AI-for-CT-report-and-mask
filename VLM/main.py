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


#LLM report generation

llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

for p in llm.parameters():
    p.requires_grad = False

#Diemensions, these will be used for the projection layers and the fusion module
vision_dim = vision_features.shape[1]
text_dim = text_encoder.config.hidden_size
hidden_dim = 512
llm_dim = llm.config.n_embd


#Projection layer
image_projection = nn.Linear(vision_dim, hidden_dim).to(device)
text_projection = nn.Linear(text_dim, hidden_dim).to(device)
llm_projection = nn.Linear(llm_dim, hidden_dim).to(device)

#Text encoder
def encode_text(text_list):
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = text_encoder(**inputs)
    
    #CLS token representation, summary of sentence
    return outputs.last_hidden_state[:, 0, :]