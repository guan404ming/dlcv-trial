"""
Inference script for answer normalization.
"""

from .ml.normalizer import AnswerNormalizer
from .lm.normalizer import lm_based_normalize_answer_sync

_normalizer = None


def ml_based_normalize_answer(
    freeform_answer, category, question=None, model_path="checkpoints"
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

    result = lm_based_normalize_answer_sync(answer, category)
    return {
        "normalized_value": result.normalized_value,
        "reasoning": result.reasoning,
    }
