"""
Agentic Answer Normalizer using local lang model with Transformers.

Uses structured I/O and tool calling for robust answer normalization.
"""

import sys
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Add project root to path to import models
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import RESPONSE_MODEL_MAP


class NormalizedAnswer(BaseModel):
    """Normalized answer with reasoning."""

    normalized_value: str = Field(
        description="The normalized answer value (number, word, or direction)"
    )
    reasoning: str = Field(description="Brief reasoning for the normalization")


class LocalNormalizer:
    """Local normalizer using Qwen3-VL model."""

    def __init__(self):
        self.model = None
        self.processor = None

    def _load_model(self):
        """Lazy load the model."""
        if self.model is not None:
            return

        print("Loading model locally...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        print("Model loaded.")

    def normalize(self, answer: str, category: str) -> NormalizedAnswer:
        """
        Normalize answer using the local model.
        """
        self._load_model()

        # Prepare schema
        if category in RESPONSE_MODEL_MAP:
            ResponseModel = RESPONSE_MODEL_MAP[category]
            schema = ResponseModel.model_json_schema()

            prompt = f"""User Answer: {answer}

Task: Normalize the "User Answer" based on the category "{category}".

Please provide your answer in JSON format matching this schema:
{json.dumps(schema, indent=2)}

Respond with valid JSON only. Include "reasoning" and "normalized_answer"."""
        else:
            # Fallback prompt if category not found (shouldn't happen given usage)
            prompt = f"""User Answer: {answer}

Normalize the user answer. Provide a JSON with "normalized_answer" and "reasoning" fields."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Generate
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse output
        return self._parse_output(model_output, category)

    def _parse_output(self, output: str, category: str) -> NormalizedAnswer:
        """Parse the JSON output from the model."""
        try:
            # Extract JSON
            # First try to find markdown code block
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Fallback to finding first { and last }
                json_start = output.find("{")
                json_end = output.rfind("}") + 1

                if json_start != -1 and json_end > json_start:
                    json_str = output[json_start:json_end]
                else:
                    raise ValueError("No JSON found in output")

            parsed_data = json.loads(json_str)

            # Handle case where model outputs schema-like structure with values in 'properties'
            if "properties" in parsed_data and isinstance(parsed_data["properties"], dict):
                props = parsed_data["properties"]
                if "reasoning" in props or "normalized_answer" in props:
                    parsed_data = props

            reasoning = parsed_data.get("reasoning", "No reasoning provided")
            norm_val = parsed_data.get("normalized_answer", "")

            if isinstance(norm_val, (int, float)):
                norm_val = str(norm_val)

            return NormalizedAnswer(
                normalized_value=norm_val,
                reasoning=reasoning,
            )

        except Exception as e:
            print(f"Error parsing model output: {e}")
            print(f"Raw output: {output}")
            # Fallback
            return NormalizedAnswer(
                normalized_value=output.strip(),
                reasoning=f"Failed to parse structured output. Raw: {output[:100]}...",
            )


# Global instance
_normalizer_instance = None


def agentic_normalize_answer_sync(
    answer: str, category: str
) -> NormalizedAnswer:
    """
    Synchronous version of agentic_normalize_answer using local model.
    """
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = LocalNormalizer()

    return _normalizer_instance.normalize(answer, category)
