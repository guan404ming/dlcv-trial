import argparse
import json
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from tools import visualize_masks_on_image


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

    # Load model with 8-bit quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
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

        # Prepare messages with annotated RGB image and depth image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": annotated_image},
                    {"type": "image", "image": depth_image},
                    {"type": "text", "text": question_text},
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
        freeform_answer = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # For normalized answer, you need to implement extraction logic based on category
        normalized_answer = freeform_answer  # Placeholder - implement extraction logic

        results.append({"id": sample["id"], "normalized_answer": normalized_answer})

        print(f"Index {idx}: {freeform_answer}")

    # Save results
    output_path = Path(args.json_path).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
