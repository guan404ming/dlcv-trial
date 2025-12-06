"""
Training script for RoBERTa-based answer normalizer.
"""

import argparse
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from pathlib import Path

from model import AnswerNormalizerModel, create_tokenizer
from utils.dataset import create_dataloaders


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct_tokens += ((predictions == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        total_loss += loss.item()
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct_tokens / total_tokens:.4f}",
            }
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct_tokens += ((predictions == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            total_loss += loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct_tokens / total_tokens:.4f}",
                }
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train RoBERTa-based answer normalizer"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train_answer_pairs.json",
        help="Path to training data JSON",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/val_answer_pairs.json",
        help="Path to validation data JSON",
    )
    parser.add_argument(
        "--model_name", type=str, default="roberta-base", help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="answer_normalizer/checkpoints",
        help="Output directory for model checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum sequence length"
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = create_tokenizer(args.model_name)

    # Create dataloaders
    print(f"\nLoading data...")
    print(f"  Train: {args.train_path}")
    print(f"  Val: {args.val_path}")
    train_loader, val_loader = create_dataloaders(
        args.train_path,
        args.val_path,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print(f"\nInitializing model: {args.model_name}")
    model = AnswerNormalizerModel(model_name=args.model_name)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'=' * 60}\n")

    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"{'-' * 60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = output_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                model_path,
            )

            # Save tokenizer
            tokenizer.save_pretrained(output_path / "tokenizer")

            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")

    # Save training history
    history_path = Path(args.output_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
