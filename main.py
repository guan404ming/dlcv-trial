import argparse
import json
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from pydantic import ValidationError
from tools import visualize_masks_on_image
from models import RESPONSE_MODEL_MAP


def main():
    parser = argparse.ArgumentParser(description="Process warehouse samples with Qwen")
    parser.add_argument("--json_path", type=str, default="data/val.json")
    parser.add_argument("--image_folder", type=str, default="data/val/images")
    parser.add_argument("--depth_folder", type=str, default="data/val/depths")
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    # Load JSON data
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # Load model with 8-bit quantization and CPU offloading
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    # Process each index
    results = []
    for idx in args.indices:
        sample = data[idx]
        category = sample.get("category", "")

        print(f"\nProcessing index {idx} (ID: {sample['id']}, Category: {category})...")

        # Load images
        image_name = sample["image"]
        base_name = Path(image_name).stem
        rgb_image = Image.open(Path(args.image_folder) / image_name)
        depth_image = Image.open(Path(args.depth_folder) / f"{base_name}_depth.png")

        # Visualize masks on RGB image
        masks = sample.get("rle", [])
        if masks:
            annotated_image = visualize_masks_on_image(rgb_image, masks)
        else:
            annotated_image = rgb_image

        # Extract question
        question = next(
            conv["value"] for conv in sample["conversations"] if conv["from"] == "human"
        )
        question_text = question.replace("<image>", "").strip()

        # Create structured prompt based on category
        if category in RESPONSE_MODEL_MAP:
            ResponseModel = RESPONSE_MODEL_MAP[category]
            schema = ResponseModel.model_json_schema()
            structured_prompt = f"""{question_text}

Please provide your answer in JSON format matching this schema:
{json.dumps(schema, indent=2)}

Respond with valid JSON only."""
        else:
            structured_prompt = question_text

        # Prepare messages with annotated RGB image and depth image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": annotated_image},
                    {"type": "image", "image": depth_image},
                    {"type": "text", "text": structured_prompt},
                ],
            }
        ]

        # Generate answer
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse with Pydantic model
        reasoning = ""
        normalized_answer = model_output
        freeform_answer = model_output

        if category in RESPONSE_MODEL_MAP:
            try:
                # Extract JSON from output
                json_start = model_output.find('{')
                json_end = model_output.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = model_output[json_start:json_end]
                    parsed_data = json.loads(json_str)

                    # Validate with Pydantic
                    ResponseModel = RESPONSE_MODEL_MAP[category]
                    validated = ResponseModel(**parsed_data)

                    reasoning = validated.reasoning
                    normalized_answer = str(validated.normalized_answer)
                    freeform_answer = validated.freeform_answer
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"  Warning: Failed to parse structured output: {e}")

        results.append({
            "id": sample["id"],
            "category": category,
            "reasoning": reasoning,
            "normalized_answer": normalized_answer,
            "freeform_answer": freeform_answer,
        })

        print(f"  Model Output: {model_output}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Normalized Answer: {normalized_answer}")
        print(f"  Freeform Answer: {freeform_answer}")

        if "normalized_answer" in sample:
            print(f"  Ground Truth: {sample['normalized_answer']}")

    # Save results
    output_path = Path(args.json_path).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
