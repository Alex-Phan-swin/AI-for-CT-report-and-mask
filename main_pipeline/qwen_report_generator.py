import threading
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_PATH = r"C:/AI/models/Qwen2.5-VL-3B-Instruct"
USE_LOCAL_MODEL = Path(MODEL_PATH).exists()


# =========================
# Progress Bar
# =========================
class ProgressBarCriteria(StoppingCriteria):
    def __init__(self, max_new_tokens):
        self.pbar = tqdm(total=max_new_tokens, desc="Generating Report")
        self.current = 0

    def __call__(self, input_ids, scores, **kwargs):
        new_current = input_ids.shape[1]
        if new_current > self.current:
            self.pbar.update(new_current - self.current)
            self.current = new_current
        return False


# =========================
# Loading animation
# =========================
def _loading_bar(stop_event):
    with tqdm(total=100, desc="Loading model") as pbar:
        while not stop_event.is_set():
            pbar.update(1)
            if pbar.n >= 100:
                pbar.reset()
            time.sleep(0.2)


# =========================
# MAIN GENERATOR
# =========================
class MedicalImageReportGenerator:

    def __init__(self):
        stop_event = threading.Event()
        t = threading.Thread(target=_loading_bar, args=(stop_event,), daemon=True)
        t.start()

        try:
            model_id = MODEL_PATH if USE_LOCAL_MODEL else "Qwen/Qwen2.5-VL-3B-Instruct"

            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            except RuntimeError:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

        finally:
            stop_event.set()
            t.join(timeout=2)

        self.processor = AutoProcessor.from_pretrained(model_id)

        print("Model ready.")

    def generate_report(self, image_path, modality="auto", max_new_tokens=256):

        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768))

        temp_path = Path("temp_input.jpg")
        img.save(temp_path, quality=95)

        prompt = f"""
You are a medical imaging assistant.

Task:
Generate a structured radiology-style report based ONLY on visible features.

Modality: {modality}

Rules:
- Do NOT guess diagnosis
- Only describe visible evidence
- If uncertain, say so explicitly
- Keep grounded in the image

Output:
- Observations
- Possible Findings
- Impression (cautious)
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(temp_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(next(self.model.parameters()).device)

        progress = ProgressBarCriteria(max_new_tokens)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([progress]),
        )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output)
        ]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
        )[0]