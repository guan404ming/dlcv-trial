"""
Unified demo script for answer normalizer.

Tests ML-based, LM-based, or both approaches on validation dataset with metrics.
"""

import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

from .main import AnswerNormalizer, lm_based_normalize_answer


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


def run_evaluation_ml(examples, normalizer, verbose=False):
    """
    Run evaluation using ML-based normalizer.

    Args:
        examples: List of examples to evaluate
        normalizer: AnswerNormalizer instance
        verbose: Whether to print detailed output for each example

    Returns:
        Dictionary with metrics
    """
    results = {
        'total': 0,
        'correct': 0,
        'errors': 0,
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0}),
        'total_time': 0.0,
        'times': []
    }

    for i, ex in enumerate(examples, 1):
        category = ex['category']
        question = ex.get('question', '')
        answer = ex['freeform_answer']
        expected = normalize_value(ex['normalized_answer'])

        results['total'] += 1
        results['by_category'][category]['total'] += 1

        if verbose:
            print(f"\n{'='*80}")
            print(f"Example {i}/{len(examples)} [{category}]")
            print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
            print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
            print(f"Expected: {expected}")

        try:
            start_time = time.time()
            predicted = normalizer.normalize(answer, category, question)
            elapsed_time = time.time() - start_time

            results['times'].append(elapsed_time)
            results['total_time'] += elapsed_time

            predicted = normalize_value(predicted)
            is_correct = predicted == expected

            if is_correct:
                results['correct'] += 1
                results['by_category'][category]['correct'] += 1

            if verbose:
                print(f"Predicted: {predicted}")
                print(f"Time: {elapsed_time:.3f}s")
                print(f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            else:
                status = '✓' if is_correct else '✗'
                print(f"[ML] [{i}/{len(examples)}] {status} {category} ({elapsed_time:.2f}s)", end='\r')

        except Exception as e:
            results['errors'] += 1
            results['by_category'][category]['errors'] += 1

            if verbose:
                print(f"Error: {e}")
            else:
                print(f"[ML] [{i}/{len(examples)}] ✗ {category} (ERROR)", end='\r')

    if not verbose:
        print()  # New line after progress

    return results


def run_evaluation_lm(examples, verbose=False):
    """
    Run evaluation using LM-based normalizer.

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
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0}),
        'total_time': 0.0,
        'times': []
    }

    for i, ex in enumerate(examples, 1):
        category = ex['category']
        question = ex.get('question', '')
        answer = ex['freeform_answer']
        expected = normalize_value(ex['normalized_answer'])

        results['total'] += 1
        results['by_category'][category]['total'] += 1

        if verbose:
            print(f"\n{'='*80}")
            print(f"Example {i}/{len(examples)} [{category}]")
            print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
            print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
            print(f"Expected: {expected}")

        try:
            start_time = time.time()
            result = lm_based_normalize_answer(
                answer=answer,
                category=category
            )
            elapsed_time = time.time() - start_time

            results['times'].append(elapsed_time)
            results['total_time'] += elapsed_time

            predicted = normalize_value(result['normalized_value'])
            is_correct = predicted == expected

            if is_correct:
                results['correct'] += 1
                results['by_category'][category]['correct'] += 1

            if verbose:
                print(f"Predicted: {predicted}")
                print(f"Reasoning: {result['reasoning'][:150]}..." if len(result['reasoning']) > 150 else f"Reasoning: {result['reasoning']}")
                print(f"Time: {elapsed_time:.3f}s")
                print(f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            else:
                status = '✓' if is_correct else '✗'
                print(f"[LM] [{i}/{len(examples)}] {status} {category} ({elapsed_time:.2f}s)", end='\r')

        except Exception as e:
            results['errors'] += 1
            results['by_category'][category]['errors'] += 1

            if verbose:
                print(f"Error: {e}")
            else:
                print(f"[LM] [{i}/{len(examples)}] ✗ {category} (ERROR)", end='\r')

    if not verbose:
        print()  # New line after progress

    return results


def print_metrics(results, method_name):
    """Print evaluation metrics."""
    print("\n" + "="*80)
    print(f"{method_name.upper()} EVALUATION RESULTS")
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

    # Timing metrics
    if results['times']:
        avg_time = results['total_time'] / len(results['times'])
        min_time = min(results['times'])
        max_time = max(results['times'])
        print(f"\nTiming Performance:")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Average per sample: {avg_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Throughput: {len(results['times']) / results['total_time']:.2f} samples/sec")

    # Per-category metrics
    print("\nPer-Category Performance:")
    for category in sorted(results['by_category'].keys()):
        stats = results['by_category'][category]
        cat_total = stats['total']
        cat_correct = stats['correct']
        cat_errors = stats['errors']
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0

        # Determine status
        if cat_accuracy >= 95:
            status = "✓"
        elif cat_accuracy >= 80:
            status = "⚠"
        else:
            status = "✗"

        print(f"  {status} {category:12s}: {cat_correct:3d}/{cat_total:3d} correct ({cat_accuracy:5.2f}%) | {cat_errors} errors")

    print("="*80)


def print_comparison(ml_results, lm_results):
    """Print comparison between ML and LM approaches."""
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)

    ml_acc = (ml_results['correct'] / ml_results['total'] * 100) if ml_results['total'] > 0 else 0
    lm_acc = (lm_results['correct'] / lm_results['total'] * 100) if lm_results['total'] > 0 else 0

    print("\nAccuracy Comparison:")
    print(f"  ML-based: {ml_results['correct']}/{ml_results['total']} = {ml_acc:.2f}%")
    print(f"  LM-based: {lm_results['correct']}/{lm_results['total']} = {lm_acc:.2f}%")
    print(f"  Winner: {'ML-based' if ml_acc > lm_acc else 'LM-based' if lm_acc > ml_acc else 'TIE'}")

    # Speed comparison
    if ml_results['times'] and lm_results['times']:
        ml_avg = ml_results['total_time'] / len(ml_results['times'])
        lm_avg = lm_results['total_time'] / len(lm_results['times'])
        speedup = lm_avg / ml_avg if ml_avg > 0 else 0

        print("\nSpeed Comparison:")
        print(f"  ML-based avg: {ml_avg:.3f}s per sample")
        print(f"  LM-based avg: {lm_avg:.3f}s per sample")
        print(f"  Speedup: ML is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than LM")
        print(f"  ML throughput: {len(ml_results['times']) / ml_results['total_time']:.2f} samples/sec")
        print(f"  LM throughput: {len(lm_results['times']) / lm_results['total_time']:.2f} samples/sec")

    print("\nBy Category:")
    print("-" * 80)
    print(f"  {'Category':12s} | {'ML Accuracy':12s} | {'LM Accuracy':12s} | {'Winner':10s}")
    print("-" * 80)

    all_categories = set(ml_results['by_category'].keys()) | set(lm_results['by_category'].keys())
    for category in sorted(all_categories):
        ml_stats = ml_results['by_category'].get(category, {'total': 0, 'correct': 0})
        lm_stats = lm_results['by_category'].get(category, {'total': 0, 'correct': 0})

        ml_cat_acc = (ml_stats['correct'] / ml_stats['total'] * 100) if ml_stats['total'] > 0 else 0
        lm_cat_acc = (lm_stats['correct'] / lm_stats['total'] * 100) if lm_stats['total'] > 0 else 0

        winner = 'ML' if ml_cat_acc > lm_cat_acc else 'LM' if lm_cat_acc > ml_cat_acc else 'TIE'

        print(f"  {category:12s} | {ml_cat_acc:10.2f}% | {lm_cat_acc:10.2f}% | {winner:10s}")

    print("="*80)


def main():
    """Run demo on validation dataset."""
    parser = argparse.ArgumentParser(description='Test answer normalizer on validation data')
    parser.add_argument('--val_path', type=str,
                       default='data/val_answer_pairs.json',
                       help='Path to validation JSON file')
    parser.add_argument('--method', type=str, default='all',
                       choices=['ml', 'lm', 'all'],
                       help='Normalization method to test (default: all)')
    parser.add_argument('--model_path', type=str,
                       default='/home/gmchiu/Documents/Github/dlcv-trial/answer_normalizer/checkpoints',
                       help='Path to ML model checkpoint (for ml method)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples to test (default: all)')
    parser.add_argument('--category', type=str, default=None,
                       choices=['count', 'distance', 'left_right', 'mcq'],
                       help='Filter by category (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output for each example')

    args = parser.parse_args()

    print("="*80)
    print("ANSWER NORMALIZER DEMO")
    print("="*80)

    # Load validation data
    print(f"\nLoading validation data from: {args.val_path}")
    examples = load_validation_data(args.val_path, args.limit, args.category)
    print(f"Loaded {len(examples)} examples")

    if args.category:
        print(f"Category filter: {args.category}")

    # Initialize normalizers based on method
    ml_normalizer = None
    if args.method in ['ml', 'all']:
        model_path = Path(args.model_path)
        if not (model_path / "best_model.pt").exists():
            print(f"\nWarning: ML model not found at {model_path}")
            print("Skipping ML-based evaluation. Train the model first using ml/train.py")
            if args.method == 'ml':
                return
        else:
            print(f"\nLoading ML model from: {model_path}")
            ml_normalizer = AnswerNormalizer(model_path)
            print("ML model loaded successfully!")

    # Run evaluations
    ml_results = None
    lm_results = None

    if args.method in ['ml', 'all'] and ml_normalizer:
        print("\nRunning ML-based evaluation...")
        ml_results = run_evaluation_ml(examples, ml_normalizer, verbose=args.verbose)

    if args.method in ['lm', 'all']:
        print("\nRunning LM-based evaluation...")
        lm_results = run_evaluation_lm(examples, verbose=args.verbose)

    # Print results
    if ml_results:
        print_metrics(ml_results, "ML-based")

    if lm_results:
        print_metrics(lm_results, "LM-based")

    if ml_results and lm_results:
        print_comparison(ml_results, lm_results)


if __name__ == "__main__":
    main()
