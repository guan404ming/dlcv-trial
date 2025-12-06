"""
Demo script for agentic answer normalizer.

Tests the agent on validation dataset with metrics and statistics.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict import agentic_normalize_answer


def normalize_value(value):
    """Normalize a value for comparison."""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def load_validation_data(val_path: str, limit: int = None, category: str = None):
    """
    Load validation data from JSON file.

    Args:
        val_path: Path to validation JSON file
        limit: Maximum number of samples to load (None for all)
        category: Filter by category (None for all categories)

    Returns:
        List of validation examples
    """
    with open(val_path, 'r') as f:
        data = json.load(f)

    # Filter by category if specified
    if category:
        data = [ex for ex in data if ex.get('category') == category]

    # Limit samples if specified
    if limit:
        data = data[:limit]

    return data


def run_evaluation(examples, verbose=False):
    """
    Run evaluation on examples and compute metrics.

    Args:
        examples: List of examples to evaluate
        verbose: Whether to print detailed output for each example

    Returns:
        Dictionary with metrics
    """
    results = {
        'total': 0,
        'correct': 0,
        'errors': 0,
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})
    }

    for i, ex in enumerate(examples, 1):
        category = ex['category']
        question = ex['question']
        answer = ex['freeform_answer']
        expected = normalize_value(ex['normalized_answer'])

        results['total'] += 1
        results['by_category'][category]['total'] += 1

        if verbose:
            print(f"\n{'='*80}")
            print(f"Example {i}/{len(examples)} [{category}]")
            print(f"Question: {question[:100]}...")
            print(f"Answer: {answer[:100]}...")
            print(f"Expected: {expected}")

        try:
            result = agentic_normalize_answer(
                answer=answer,
                category=category
            )

            predicted = normalize_value(result['normalized_value'])
            is_correct = predicted == expected

            if is_correct:
                results['correct'] += 1
                results['by_category'][category]['correct'] += 1

            if verbose:
                print(f"Predicted: {predicted}")
                print(f"Reasoning: {result['reasoning'][:150]}...")
                print(f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            else:
                # Show progress in non-verbose mode
                status = '✓' if is_correct else '✗'
                print(f"[{i}/{len(examples)}] {status} {category}", end='\r')

        except Exception as e:
            results['errors'] += 1
            results['by_category'][category]['errors'] += 1

            if verbose:
                print(f"Error: {e}")
            else:
                print(f"[{i}/{len(examples)}] ✗ {category} (ERROR)", end='\r')

    if not verbose:
        print()  # New line after progress

    return results


def print_metrics(results):
    """Print evaluation metrics."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Overall metrics
    total = results['total']
    correct = results['correct']
    errors = results['errors']
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\nOverall Performance:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Wrong: {total - correct - errors}")
    print(f"  Errors: {errors}")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Per-category metrics
    print("\nPer-Category Performance:")
    for category in sorted(results['by_category'].keys()):
        stats = results['by_category'][category]
        cat_total = stats['total']
        cat_correct = stats['correct']
        cat_errors = stats['errors']
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0

        print(f"  {category:12s}: {cat_correct:3d}/{cat_total:3d} correct ({cat_accuracy:5.2f}%) | {cat_errors} errors")

    print("="*80)


def main():
    """Run demo on validation dataset."""
    parser = argparse.ArgumentParser(description='Test agentic answer normalizer on validation data')
    parser.add_argument('--val_path', type=str,
                       default='data/val_answer_pairs.json',
                       help='Path to validation JSON file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples to test (default: all)')
    parser.add_argument('--category', type=str, default=None,
                       choices=['count', 'distance', 'left_right', 'mcq'],
                       help='Filter by category (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output for each example')

    args = parser.parse_args()

    print("Agentic Answer Normalizer - Validation Test")
    print("="*80)

    # Load validation data
    print(f"\nLoading validation data from: {args.val_path}")
    examples = load_validation_data(args.val_path, args.limit, args.category)
    print(f"Loaded {len(examples)} examples")

    if args.category:
        print(f"Category filter: {args.category}")

    # Run evaluation
    print("\nRunning evaluation...")
    results = run_evaluation(examples, verbose=args.verbose)

    # Print metrics
    print_metrics(results)


if __name__ == "__main__":
    main()
