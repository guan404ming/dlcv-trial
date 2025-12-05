import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import argparse


def load_data(json_path):
    """Load question-category pairs from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    questions = [item['question'] for item in data]
    categories = [item['category'] for item in data]
    return questions, categories


def clean_question(question):
    """Clean questions while keeping special tokens."""
    # Keep <image> and <mask> tokens as they contain important information
    return question.strip()


def train_model(train_questions, train_categories):
    """Train a simple TF-IDF + Logistic Regression classifier."""
    # Clean questions
    train_questions = [clean_question(q) for q in train_questions]

    # Create pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        ))
    ])

    # Train
    print(f"Training on {len(train_questions)} samples...")
    model.fit(train_questions, train_categories)

    return model


def evaluate_model(model, test_questions, test_categories):
    """Evaluate model on test data."""
    # Clean questions
    test_questions = [clean_question(q) for q in test_questions]

    # Predict
    predictions = model.predict(test_questions)

    # Calculate metrics
    accuracy = accuracy_score(test_categories, predictions)
    report = classification_report(test_categories, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    return accuracy, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train_pairs.json')
    parser.add_argument('--val_path', type=str, default='data/val_pairs.json')
    parser.add_argument('--model_path', type=str, default='models/question_classifier.pkl')
    args = parser.parse_args()

    # Load training data
    print("Loading training data...")
    train_questions, train_categories = load_data(args.train_path)
    print(f"Loaded {len(train_questions)} training samples")

    # Train model
    model = train_model(train_questions, train_categories)

    # Evaluate on training data
    print("\n=== Training Set Performance ===")
    evaluate_model(model, train_questions, train_categories)

    # Evaluate on validation data
    print("\n=== Validation Set Performance ===")
    val_questions, val_categories = load_data(args.val_path)
    evaluate_model(model, val_questions, val_categories)

    # Save model
    import os
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    with open(args.model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {args.model_path}")


if __name__ == '__main__':
    main()
