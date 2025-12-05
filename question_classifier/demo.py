import json
from pathlib import Path
from inference import predict

def main():
    # Load test data
    test_path = Path(__file__).parent.parent / 'data' / 'test.json'
    with open(test_path) as f:
        data = json.load(f)

    print(f"Processing {len(data)} samples...")

    # Process each sample
    results = []
    for item in data:
        # Extract question from conversations
        question = None
        for conv in item.get('conversations', []):
            if conv.get('from') == 'human':
                question = conv.get('value')
                break

        if not question:
            continue

        # Predict category
        category = predict(question)

        # Store result
        results.append({
            'id': item['id'],
            'image': item['image'],
            'question': question,
            'predicted_category': category
        })

    # Save results
    output_path = Path(__file__).parent / 'result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")

    # Show category distribution
    from collections import Counter
    counts = Counter(r['predicted_category'] for r in results)
    print("\nCategory distribution:")
    for cat, count in sorted(counts.items()):
        print(f"  {cat:12s}: {count}")

if __name__ == '__main__':
    main()
