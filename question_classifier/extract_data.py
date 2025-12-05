import json
import argparse
import os


def extract_pairs(data):
    results = []

    for item in data:
        conv = item.get("conversations", [])
        question = None

        for entry in conv:
            if entry.get("from") == "human":
                question = entry.get("value")
                break

        category = item.get("category")

        if question:
            results.append(
                {
                    "question": question,
                    "category": category,
                }
            )

    return results


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def dedup(old_list, new_list):
    """Dedup based on (question, category)."""
    seen = set((item["question"], item["category"]) for item in old_list)

    merged = old_list[:]
    added = 0

    for item in new_list:
        key = (item["question"], item["category"])
        if key not in seen:
            merged.append(item)
            seen.add(key)
            added += 1

    print(f"Added {added} new unique entries.")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="data/train.json")
    parser.add_argument(
        "--save_path", type=str, default="data/question_category_pairs.json"
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        dataset = json.load(f)

    new_pairs = extract_pairs(dataset)

    existing_pairs = load_existing(args.save_path)

    merged = dedup(existing_pairs, new_pairs)

    with open(args.save_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Total after dedup: {len(merged)}")
