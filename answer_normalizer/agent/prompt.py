"""
Prompt templates for agentic answer normalizer.

Contains category-specific prompts and few-shot examples for answer normalization.
"""

# System prompt for the agent
SYSTEM_PROMPT = """You are an expert answer normalizer for warehouse spatial intelligence questions.

Your task is to extract the normalized answer from freeform text based on the question category:

- **count**: Extract the number of objects (pallets, etc.) as a numeric string or word
- **distance**: Extract the distance value in meters as a decimal number
- **left_right**: Extract spatial relationship (left, right, front, behind, above, below)
- **mcq**: Extract the region number that is the correct answer

Be precise and confident in your extraction. Always provide reasoning for your normalization."""


# Category-specific few-shot examples
COUNT_EXAMPLES = """
Examples for COUNT questions:

Example 1:
Answer: From the image's perspective, the buffer region [Region 1] is the leftmost buffer region. You can find pallet [Region 6] inside the buffer region [Region 1]. It means that there is one pallet in the buffer zone [Region 1].
Normalized: 1

Example 2:
Answer: The buffer region [Region 0] is the leftmost buffer region among all the buffer regions. You can find pallets [Region 3] [Region 6] [Region 8] [Region 12] inside the buffer region [Region 0]. Therefore, the buffer region [Region 0] holds a total of four pallets.
Normalized: 4

Example 3:
Answer: From this viewpoint, the buffer region [Region 0] is the rightmost buffer region. I see pallets [Region 6] [Region 8] [Region 9] in the buffer region [Region 0]. Thus, there are three pallets in the buffer area [Region 0].
Normalized: 3
"""


DISTANCE_EXAMPLES = """
Examples for DISTANCE questions:

Example 1:
Answer: The pallet [Region 0] is 8.66 meters away from the pallet [Region 1].
Normalized: 8.66

Example 2:
Answer: Among the pallets, [Region 4] is the rightmost pallet. There is a distance of 13.65 meters between the pallet [Region 4] and the shelf [Region 6].
Normalized: 13.65

Example 3:
Answer: From this viewpoint, [Region 8] is the rightmost pallet. [Region 0] is the leftmost buffer region from this viewpoint. The distance from the pallet [Region 8] to the buffer region [Region 0] is 2.49 meters.
Normalized: 2.49
"""


LEFT_RIGHT_EXAMPLES = """
Examples for LEFT_RIGHT (spatial relationship) questions:

Example 1:
Answer: The pallet [Region 0] is to the left of the pallet [Region 1].
Normalized: left

Example 2:
Answer: The pallet [Region 0] is to the left of the pallet [Region 1].
Normalized: left

Example 3:
Answer: The pallet [Region 0] is situated on the right of the pallet [Region 1].
Normalized: right

Example 4:
Answer: Looking from this angle, the pallet [Region 0] is to the right of the pallet [Region 1].
Normalized: right
"""


MCQ_EXAMPLES = """
Examples for MCQ (multiple choice) questions:

Example 1:
Answer: Among all the transporters, the transporter [Region 0] is empty. The pallet [Region 5] is the closest to transporter [Region 0].
Normalized: 5

Example 2:
Answer: Among all the transporters, the transporter [Region 0] is empty. The pallet [Region 5] is the closest to transporter [Region 0], so it is the most suitable choice for automated picking.
Normalized: 5

Example 3:
Answer: Among all the transporters, the transporter [Region 11] is empty. Given that pallet [Region 5] is the nearest to transporter [Region 11], it is the optimal choice to pick up.
Normalized: 5

Example 4:
Answer: The buffer zone [Region 1] is the shortest distance from the shelf [Region 3].
Normalized: 1
"""


def get_category_examples(category: str) -> str:
    """
    Get few-shot examples for a specific category.

    Args:
        category: Question category (count, distance, left_right, mcq)

    Returns:
        Few-shot examples as a string
    """
    examples_map = {
        "count": COUNT_EXAMPLES,
        "distance": DISTANCE_EXAMPLES,
        "left_right": LEFT_RIGHT_EXAMPLES,
        "mcq": MCQ_EXAMPLES,
    }

    return examples_map.get(category, "")


def build_normalization_prompt(answer: str, category: str) -> str:
    """
    Build a complete prompt for answer normalization.

    Args:
        answer: The freeform answer to normalize
        category: The question category

    Returns:
        Complete prompt with examples and instructions
    """
    examples = get_category_examples(category)

    prompt = f"""{examples}

Now normalize this answer:

Answer: {answer}

Category: {category}

Extract and normalize the answer from the freeform text above.
Provide the normalized value and brief reasoning."""

    return prompt


# Category-specific extraction instructions
CATEGORY_INSTRUCTIONS = {
    "count": """
For COUNT questions:
- Extract the numeric count of objects (pallets, etc.)
- Can be a digit (1, 2, 3) or word (one, two, three)
- Look for phrases like "there are X pallets", "holds a total of X", "X pallets are"
- Return just the number or number word
""",

    "distance": """
For DISTANCE questions:
- Extract the distance value in meters
- Return as a decimal number (e.g., 8.66, 13.65, 2.49)
- Look for patterns like "X meters", "distance of X", "X meters away"
- Keep the decimal precision from the original answer
""",

    "left_right": """
For LEFT_RIGHT questions:
- Extract the spatial relationship word
- Valid values: left, right
- Look for phrases like "to the left of", "on the right", "situated left"
- Return just the single relationship word
""",

    "mcq": """
For MCQ questions:
- Extract the region number that is the answer
- Look for mentions like "pallet [Region X]", "buffer [Region X]"
- Find which region is described as the answer (closest, best, nearest, etc.)
- Return just the numeric region identifier
""",
}
