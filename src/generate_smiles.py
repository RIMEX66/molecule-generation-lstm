from smiles_tokenizer import decode_ids
from model import SmilesLSTM
import torch.nn.functional as F
import torch
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


VOCAB_FILE = "models/vocab.json"
MODEL_FILE = "models/lstm_best.pt"
OUT_FILE = "submissions/generated_raw.txt"

NUM_SMILES = 20000
MAX_LEN = 120
TEMPERATURE = 1.0


def sample_next_token(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_one(model, token_to_id, id_to_token, device, max_len=120, temperature=1.0):
    model.eval()

    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]

    ids = [bos_id]
    hidden = None

    with torch.no_grad():
        current = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits, hidden = model(current, hidden)

            next_logits = logits[0, -1, :]

            # Do not sample padding or BOS tokens during generation.
            next_logits[token_to_id["<pad>"]] = -float("inf")
            next_logits[token_to_id["<bos>"]] = -float("inf")

            next_id = sample_next_token(next_logits, temperature)

            if next_id == eos_id:
                break

            ids.append(next_id)

            current = torch.tensor(
                [[next_id]], dtype=torch.long, device=device)

    return decode_ids(ids, id_to_token)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    with open(VOCAB_FILE, "r") as f:
        vocab_data = json.load(f)

    vocab = vocab_data["vocab"]
    token_to_id = vocab_data["token_to_id"]
    id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}

    vocab_size = len(vocab)

    checkpoint = torch.load(MODEL_FILE, map_location=device)

    config = checkpoint["config"]

    model = SmilesLSTM(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {MODEL_FILE}")
    print(f"Generating {NUM_SMILES} SMILES")
    print(f"Temperature: {TEMPERATURE}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    generated = []

    for i in range(NUM_SMILES):
        smi = generate_one(
            model=model,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            device=device,
            max_len=MAX_LEN,
            temperature=TEMPERATURE,
        )

        generated.append(smi)

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{NUM_SMILES}")

    with open(OUT_FILE, "w") as f:
        for smi in generated:
            f.write(smi + "\n")

    print(f"Saved raw generated SMILES to {OUT_FILE}")

    print("\nExamples:")
    for smi in generated[:20]:
        print(smi)


if __name__ == "__main__":
    main()
