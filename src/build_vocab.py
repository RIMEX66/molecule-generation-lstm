import json
from smiles_tokenizer import build_vocabulary

TRAIN_FILE = "smiles_train.txt"
OUTPUT_FILE = "models/vocab.json"

with open(TRAIN_FILE, "r") as f:
    smiles_list = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(smiles_list)} molecules")

vocabulary, token_to_id, id_to_token = build_vocabulary(smiles_list)

with open(OUTPUT_FILE, "w") as f:
    json.dump(
        {
            "vocab": vocabulary,
            "token_to_id": token_to_id,
            "id_to_token": {str(k): v for k, v in id_to_token.items()}
        },
        f,
        indent=2
    )

print(f"Vocabulary size: {len(vocabulary)}")

# print(vocabulary)

print(f"\nSaved to {OUTPUT_FILE}")
