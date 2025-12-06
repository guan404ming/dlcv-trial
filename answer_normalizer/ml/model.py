"""
RoBERTa-based Answer Normalizer Model.

Uses token classification to identify answer spans in freeform answers.
"""

import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast


class AnswerNormalizerModel(nn.Module):
    """
    RoBERTa-based model for answer normalization.

    Uses token classification to identify the normalized answer span
    within the freeform answer text.
    """

    def __init__(self, model_name="roberta-base", num_labels=3):
        """
        Initialize the model.

        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of BIO labels (default: 3 for B, I, O)
        """
        super().__init__()

        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config = self.roberta.config

        # Dropout
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Token classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            labels: BIO labels [batch_size, seq_len] (optional, for training)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get sequence output (last hidden state)
        sequence_output = (
            outputs.last_hidden_state
        )  # [batch_size, seq_len, hidden_size]

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Get logits for each token
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.classifier.out_features), labels.view(-1)
            )

        return {"logits": logits, "loss": loss}


def create_tokenizer(model_name="roberta-base"):
    """Create RoBERTa tokenizer."""
    return RobertaTokenizerFast.from_pretrained(model_name)
