from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList
)

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch


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


class BrainCTReportGenerator:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir=r"A:\model"
    ):

        print("Loading model...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        print("Model loaded successfully!")

    def generate_report(
        self,
        image_path,
        evidence = None,
        max_new_tokens=128
    ):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
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

        # Prepare text prompt
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process image
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Progress bar
        progress = ProgressBarCriteria(max_new_tokens)

        # Generate output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([progress])
        )

        # Remove prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]