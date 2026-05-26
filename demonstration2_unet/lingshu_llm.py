import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

#https://huggingface.co/lingshu-medical-mllm/Lingshu-7B
#Model import and processor setup from lingshu 
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "lingshu-medical-mllm/Lingshu-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=r"A:\model"
)

processor = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-7B")

#Message input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": r"C:\Users\Alex\Music\project\COS40005-Computing-Technology-Project-A-H\demonstration2_unet\demo_input\TCGA_CS_4941_19960909_11.tif",
            },
            {"type": "text", 
             "text": "Analyze this brain ct scan image and provide a detailed report on any abnormalities you find or if it doesnt contain any abnormalities."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

#convert image and text into pytorch tensors for model input
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

# Convert input into tokens and decodes tokens into output text
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)