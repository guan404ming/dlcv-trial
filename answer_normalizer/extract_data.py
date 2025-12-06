"""
Extract answer normalization pairs from dataset JSON files.
"""

import json
import argparse
from pathlib import Path


def extract_answer_pairs(json_path):
    """
    Extract (question, freeform_answer, category, normalized_answer) pairs from JSON.

    Args:
        json_path: Path to dataset JSON file

    Returns:
        List of dicts with 'question', 'freeform_answer', 'category', 'normalized_answer'
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        # Skip if missing required fields
        if "freeform_answer" not in item or "normalized_answer" not in item:
            continue

        # Extract question from conversations
        question = ""
        if "conversations" in item and len(item["conversations"]) > 0:
            question = item["conversations"][0].get("value", "")
            # Remove <image> and <mask> tokens for cleaner text
            question = (
                question.replace("<image>", "").replace("<mask>", "[MASK]").strip()
            )

        pairs.append(
            {
                "question": question,
                "freeform_answer": item["freeform_answer"],
                "category": item["category"],
                "normalized_answer": item["normalized_answer"],
            }
        )

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract answer normalization pairs from dataset"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JSON file"
    )
    args = parser.parse_args()

    # Extract pairs
    print(f"Extracting answer pairs from {args.input}...")
    pairs = extract_answer_pairs(args.input)
    print(f"Extracted {len(pairs)} answer pairs")

    # Show distribution by category
    category_counts = {}
    for pair in pairs:
        cat = pair["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
