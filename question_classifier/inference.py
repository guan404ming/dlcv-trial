import pickle
from pathlib import Path

_MODEL = None


def predict(question: str) -> str:
    """
    Predict category for a question.

    Args:
        question: Question text

    Returns:
        Category: 'count', 'distance', 'left_right', or 'mcq'
    """
    global _MODEL

    if _MODEL is None:
        model_path = Path(__file__).parent / 'question_classifier.pkl'
        with open(model_path, 'rb') as f:
            _MODEL = pickle.load(f)

    return _MODEL.predict([question.strip()])[0]
