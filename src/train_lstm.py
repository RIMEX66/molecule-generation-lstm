import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SmilesLSTM
from dataset import SmilesDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# Config

TRAIN_FILE = "smiles_train.txt"
VOCAB_FILE = "models/vocab.json"
MODEL_DIR = "models"

MAX_LEN = 120
BATCH_SIZE = 512
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2

LR = 1e-3
EPOCHS = 15
TRAIN_LIMIT = None

SAVE_EVERY_EPOCH = True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    with open(VOCAB_FILE, "r") as f:
        vocab_data = json.load(f)

    vocab_size = len(vocab_data["vocab"])
    pad_idx = vocab_data["token_to_id"]["<pad>"]

    print("Vocabulary size:", vocab_size)
    print("Padding index:", pad_idx)

    dataset = SmilesDataset(
        smiles_file=TRAIN_FILE,
        vocab_file=VOCAB_FILE,
        max_len=MAX_LEN,
        limit=TRAIN_LIMIT,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    print("Training molecules:", len(dataset))
    print("Batches per epoch:", len(dataloader))

    model = SmilesLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    best_loss = float("inf")

    print("\nTraining started:")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()

        total_loss = 0.0
        total_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if batch_idx % 50 == 0:
                avg_loss = total_loss / total_batches
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f}"
                )

        avg_epoch_loss = total_loss / total_batches
        scheduler.step(avg_epoch_loss)

        elapsed = time.time() - start_time

        print("-" * 60)
        print(
            f"Epoch {epoch} finished | "
            f"Avg loss: {avg_epoch_loss:.4f} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_epoch_loss,
            "config": {
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "lr": LR,
                "train_limit": TRAIN_LIMIT,
                "vocab_size": vocab_size,
            },
        }

        if SAVE_EVERY_EPOCH:
            epoch_path = os.path.join(
                MODEL_DIR,
                f"lstm_epoch_{epoch}.pt"
            )
            torch.save(checkpoint, epoch_path)
            print(f"Saved checkpoint: {epoch_path}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(MODEL_DIR, "lstm_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        print("=" * 60)

    print("Training complete.")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
