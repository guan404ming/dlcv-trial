"""
Inference module for answer normalization using RoBERTa model.
"""

import torch
import re
from pathlib import Path
from transformers import RobertaTokenizerFast
from .model import AnswerNormalizerModel


class AnswerNormalizer:
    """
    Answer normalizer using trained RoBERTa model.
    """

    def __init__(self, model_path, device=None):
        """
        Initialize the normalizer.

        Args:
            model_path: Path to model checkpoint directory
            device: Device to use (defaults to cuda if available)
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer
        tokenizer_path = self.model_path / "tokenizer"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(str(tokenizer_path))

        # Load model
        self.model = AnswerNormalizerModel()
        checkpoint = torch.load(
            self.model_path / "best_model.pt", map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.id2label = {0: "O", 1: "B", 2: "I"}

    def normalize(self, freeform_answer, category, question=None):
        """
        Normalize a freeform answer based on its category.

        Args:
            freeform_answer: The freeform answer text
            category: The question category (mcq, distance, count, left_right)
            question: Optional question text for additional context

        Returns:
            Normalized answer string
        """
        # Create input text with optional question
        if question:
            input_text = f"{question} [SEP] {freeform_answer} [SEP] {category}"
        else:
            input_text = f"{freeform_answer} [SEP] {category}"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)

        # Extract answer span
        predictions = predictions[0].cpu().numpy()
        offset_mapping = encoding["offset_mapping"][0].numpy()

        # Find B and I tokens
        answer_tokens = []
        for pred, (start, end) in zip(predictions, offset_mapping):
            # Skip special tokens
            if start == 0 and end == 0:
                continue

            label = self.id2label[pred]
            if label in ["B", "I"]:
                # Extract text for this token from input_text
                token_text = input_text[start:end]
                answer_tokens.append(token_text)

        # Join tokens to form answer
        if answer_tokens:
            extracted_answer = "".join(answer_tokens).strip()
        else:
            # Fallback: use regex-based extraction
            extracted_answer = self._fallback_extraction(freeform_answer, category)

        # Post-process based on category
        normalized_answer = self._post_process(extracted_answer, category)

        # Special handling for count: extract explicit count phrases
        if category == "count":
            # Word to number mapping
            word_to_num = {
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
                "zero": "0",
                "no": "0",
                "eleven": "11",
                "twelve": "12",
            }

            # Look for explicit count phrases - order matters (most specific first)
            count_patterns = [
                r"(?:there (?:is|are)(?: exactly)?) (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+) pallets?",
                r"(?:holds?|has|have)(?: a)? total of (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)",
                r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) pallets? (?:has been|have been|is|are)",
                r"total of (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+) pallets?",
                r"(?:find|see) pallets.*(?:there (?:is|are)) (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)",
            ]

            for pattern in count_patterns:
                match = re.search(pattern, freeform_answer.lower())
                if match:
                    count_word = match.group(1)
                    # Convert word to number
                    if count_word in word_to_num:
                        normalized_answer = word_to_num[count_word]
                        break
                    # Check if it's already a number
                    elif count_word.isdigit():
                        normalized_answer = count_word
                        break

        return normalized_answer

    def _fallback_extraction(self, text, category):
        """
        Fallback extraction using regex when model fails.

        Args:
            text: Text to extract from
            category: Question category

        Returns:
            Extracted answer
        """
        if category == "distance":
            # Extract first number with optional decimal
            match = re.search(r"\d+\.?\d*", text)
            return match.group() if match else ""

        elif category == "count":
            # Extract first number
            match = re.search(r"\d+", text)
            if match:
                return match.group()
            # Try to find word numbers
            word_numbers = {
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
            }
            for word, num in word_numbers.items():
                if word in text.lower():
                    return num
            return ""

        elif category == "mcq":
            # Extract region number
            match = re.search(r"Region \[(\d+)\]|Region (\d+)|\[Region (\d+)\]", text)
            if match:
                return match.group(1) or match.group(2) or match.group(3)
            # Try to find any number
            match = re.search(r"\d+", text)
            return match.group() if match else ""

        elif category == "left_right":
            # Extract spatial relation
            text_lower = text.lower()
            if "left" in text_lower:
                return "left"
            elif "right" in text_lower:
                return "right"
            elif "front" in text_lower or "ahead" in text_lower:
                return "front"
            elif "behind" in text_lower or "back" in text_lower:
                return "behind"
            elif "above" in text_lower or "top" in text_lower:
                return "above"
            elif "below" in text_lower or "bottom" in text_lower:
                return "below"
            return ""

        return ""

    def _post_process(self, answer, category):
        """
        Post-process extracted answer based on category.

        Args:
            answer: Extracted answer
            category: Question category

        Returns:
            Post-processed answer
        """
        if not answer:
            return answer

        # Clean whitespace
        answer = answer.strip()

        if category == "distance":
            # Extract just the number, remove 'meters' or other units
            match = re.search(r"\d+\.?\d*", answer)
            return match.group() if match else answer

        elif category == "count":
            # Special handling: Count might need to count region mentions
            # First try to extract explicit count from extracted answer
            match = re.search(r"\d+", answer)
            if match:
                return match.group()
            # Keep word form if present
            return answer

        elif category == "mcq":
            # Extract just the number
            match = re.search(r"\d+", answer)
            return match.group() if match else answer

        elif category == "left_right":
            # Normalize to lowercase
            answer_lower = answer.lower()
            # Map to canonical forms
            if "left" in answer_lower:
                return "left"
            elif "right" in answer_lower:
                return "right"
            elif "front" in answer_lower or "ahead" in answer_lower:
                return "front"
            elif "behind" in answer_lower or "back" in answer_lower:
                return "behind"
            elif "above" in answer_lower or "top" in answer_lower:
                return "above"
            elif "below" in answer_lower or "bottom" in answer_lower:
                return "below"
            return answer

        return answer
