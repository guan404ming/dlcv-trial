"""
Demo script for answer normalizer - validates on val.json.

Shows model performance on actual validation data.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict import AnswerNormalizer


def main():
    """Run validation on val.json."""
    print("Answer Normalizer Validation")
    print("=" * 80)

    # Check if model exists
    model_path = Path(__file__).parent.parent / "checkpoints"
    if not (model_path / "best_model.pt").exists():
        print("\nError: Model not found!")
        print("Please train the model first using train.py")
        print(f"Expected path: {model_path}")
        return

    # Load validation data
    val_data_path = (
        Path(__file__).parent.parent.parent / "data" / "val_answer_pairs.json"
    )
    if not val_data_path.exists():
        print(f"\nError: Validation data not found at {val_data_path}")
        return

    print(f"\nLoading validation data from: {val_data_path}")
    with open(val_data_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples")

    # Initialize normalizer
    print(f"\nLoading model from: {model_path}")
    normalizer = AnswerNormalizer(model_path)
    print("Model loaded successfully!\n")

    # Validate on all samples
    print("Running validation...")
    print("-" * 80)

    results = {"correct": 0, "total": 0}
    category_results = {}

    for item in val_data:
        question = item.get("question", "")
        freeform = item["freeform_answer"]
        category = item["category"]
        expected = str(item["normalized_answer"])

        # Normalize with question
        predicted = normalizer.normalize(freeform, category, question)

        # Check if correct
        is_correct = predicted == expected

        # Track results
        results["total"] += 1
        if is_correct:
            results["correct"] += 1

        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}
        category_results[category]["total"] += 1
        if is_correct:
            category_results[category]["correct"] += 1

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    overall_acc = 100 * results["correct"] / results["total"]
    print(
        f"\nOverall Accuracy: {results['correct']}/{results['total']} = {overall_acc:.1f}%"
    )

    print("\nBy Category:")
    print("-" * 80)

    for cat in sorted(category_results.keys()):
        res = category_results[cat]
        acc = 100 * res["correct"] / res["total"]

        # Determine status emoji
        if acc >= 95:
            status = "✓"
        elif acc >= 80:
            status = "⚠"
        else:
            status = "✗"

        print(f"{status} {cat:15} {res['correct']:3}/{res['total']:3} = {acc:6.1f}%")

    print("=" * 80)

    # Show some failure examples
    if results["correct"] < results["total"]:
        print("\nSample Failures (first 5):")
        print("-" * 80)

        failures_shown = 0
        for item in val_data:
            if failures_shown >= 5:
                break

            question = item.get("question", "")
            freeform = item["freeform_answer"]
            category = item["category"]
            expected = str(item["normalized_answer"])
            predicted = normalizer.normalize(freeform, category, question)

            if predicted != expected:
                failures_shown += 1
                print(f"\n{failures_shown}. [{category}]")
                print(f"   Expected: {expected}")
                print(f"   Got:      {predicted}")
                if len(freeform) > 100:
                    print(f"   Answer:   {freeform[:100]}...")
                else:
                    print(f"   Answer:   {freeform}")

        print("=" * 80)


if __name__ == "__main__":
    main()
