"""
Dataset class for answer normalization.
"""

import json
import torch
from torch.utils.data import Dataset


class AnswerNormalizationDataset(Dataset):
    """
    Dataset for answer normalization task.

    Creates token classification labels (BIO tagging) for extracting
    normalized answers from freeform answers.
    """

    def __init__(self, data_path, tokenizer, max_length=256):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON file with answer pairs
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(data_path, "r") as f:
            self.data = json.load(f)

        # Label mapping
        self.label2id = {"O": 0, "B": 1, "I": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example.

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        question = item.get("question", "")
        freeform_answer = item["freeform_answer"]
        category = item["category"]
        normalized_answer = str(item["normalized_answer"])

        # Create input text: "[CLS] question [SEP] freeform_answer [SEP] category [SEP]"
        if question:
            input_text = f"{question} [SEP] {freeform_answer} [SEP] {category}"
            # Offset for finding answer in the combined text
            answer_offset = len(question) + len(" [SEP] ")
        else:
            input_text = f"{freeform_answer} [SEP] {category}"
            answer_offset = 0

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Create BIO labels
        labels = self._create_labels(
            freeform_answer,
            normalized_answer,
            encoding["offset_mapping"][0],
            answer_offset,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _create_labels(
        self, freeform_answer, normalized_answer, offset_mapping, answer_offset=0
    ):
        """
        Create BIO labels for token classification.

        Args:
            freeform_answer: Original freeform answer text
            normalized_answer: Target normalized answer
            offset_mapping: Token offset mapping from tokenizer
            answer_offset: Character offset where answer starts in the full input text

        Returns:
            List of label IDs
        """
        labels = []

        # Find normalized answer in freeform answer
        # Convert to lowercase for matching
        freeform_lower = freeform_answer.lower()
        normalized_lower = normalized_answer.lower()

        # Try to find exact match
        answer_start = freeform_lower.find(normalized_lower)

        if answer_start == -1:
            # If exact match not found, try to find partial matches
            # For numbers, try to find digit sequences
            import re

            if normalized_answer.replace(".", "").isdigit():
                # Find all numbers in text
                for match in re.finditer(r"\d+\.?\d*", freeform_answer):
                    if match.group() == normalized_answer:
                        answer_start = match.start()
                        break

        # Adjust for offset in the combined input text
        if answer_start != -1:
            answer_start += answer_offset
            answer_end = answer_start + len(normalized_answer)
        else:
            answer_end = -1

        # Create labels based on token offsets
        for i, (start, end) in enumerate(offset_mapping):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if start == 0 and end == 0:
                labels.append(-100)  # Ignore in loss
                continue

            # Check if token overlaps with answer span
            if answer_start != -1 and start < answer_end and end > answer_start:
                if start == answer_start or (i > 0 and labels[-1] == -100):
                    labels.append(self.label2id["B"])  # Beginning
                else:
                    labels.append(self.label2id["I"])  # Inside
            else:
                labels.append(self.label2id["O"])  # Outside

        return labels


def create_dataloaders(train_path, val_path, tokenizer, batch_size=16, max_length=256):
    """
    Create training and validation dataloaders.

    Args:
        train_path: Path to training data JSON
        val_path: Path to validation data JSON
        tokenizer: RoBERTa tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = AnswerNormalizationDataset(train_path, tokenizer, max_length)
    val_dataset = AnswerNormalizationDataset(val_path, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader
