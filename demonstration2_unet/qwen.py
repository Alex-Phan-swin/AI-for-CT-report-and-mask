from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

class ProgressBarCriteria(StoppingCriteria):
    def __init__(self, max_new_tokens):
        self.pbar = tqdm(total=max_new_tokens)
        self.current = 0

    def __call__(self, input_ids, scores, **kwargs):
        new_current = input_ids.shape[1]

        if new_current > self.current:
            self.pbar.update(new_current - self.current)
            self.current = new_current

        return False


# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=r"A:\model"
)

# Load processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

# Your CT image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": r"C:\Users\Alex\Music\project\COS40005-Computing-Technology-Project-A-H\demonstration2_unet\demo_input\TCGA_CS_4941_19960909_11.tif",
            },
            {
                "type": "text",
                "text": """
Generate a concise radiology report for this non-contrast brain CT scan.
Include:
- Findings
- Impression
Keep it medically concise.
"""
            },
        ],
    }
]

# Prepare inputs
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")


max_tokens = 128

progress = ProgressBarCriteria(max_tokens)

# Generate
generated_ids = model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    stopping_criteria=StoppingCriteriaList([progress])
)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text[0])