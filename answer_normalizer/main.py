"""
Inference script for answer normalization.
"""

from .ml.normalizer import AnswerNormalizer


# Global instance for easy import
_normalizer = None


def ml_based_normalize_answer(
    freeform_answer, category, question=None, model_path="answer_normalizer/checkpoints"
):
    """
    Convenience function to normalize an answer using RoBERTa model.

    Args:
        freeform_answer: The freeform answer text
        category: The question category
        question: Optional question text for additional context
        model_path: Path to model checkpoint (optional)

    Returns:
        Normalized answer string
    """
    global _normalizer

    if _normalizer is None:
        _normalizer = AnswerNormalizer(model_path)

    return _normalizer.normalize(freeform_answer, category, question)


def lm_based_normalize_answer(answer: str, category: str) -> dict:
    """
    Normalize an answer using the language model based approach.

    Uses local transformer model with structured I/O.

    Args:
        answer: The freeform answer text
        category: The question category (count, distance, left_right, mcq)

    Returns:
        Dictionary with:
            - normalized_value: The normalized answer
            - reasoning: Brief reasoning for the normalization
    """
    from .lm.normalizer import lm_based_normalize_answer_sync

    result = lm_based_normalize_answer_sync(answer, category)
    return {
        "normalized_value": result.normalized_value,
        "reasoning": result.reasoning,
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Normalize answers")
    parser.add_argument("--answer", type=str, required=True, help="Freeform answer")
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=["mcq", "distance", "count", "left_right"],
        help="Question category",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="answer_normalizer/checkpoints",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    normalizer = AnswerNormalizer(args.model_path)
    result = normalizer.normalize(args.answer, args.category)

    print(f"\nInput: {args.answer}")
    print(f"Category: {args.category}")
    print(f"Normalized: {result}\n")
