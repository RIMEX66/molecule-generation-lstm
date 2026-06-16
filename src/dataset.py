import json
import torch
from torch.utils.data import Dataset

from smiles_tokenizer import encode_smiles


class SmilesDataset(Dataset):

    def __init__(
        self,
        smiles_file,
        vocab_file,
        max_len=120,
        limit=None
    ):
        self.max_len = max_len

        with open(vocab_file, "r") as f:
            vocab_data = json.load(f)

        self.token_to_id = vocab_data["token_to_id"]

        with open(smiles_file, "r") as f:
            smiles = [line.strip() for line in f if line.strip()]

        if limit is not None:
            smiles = smiles[:limit]

        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        ids = encode_smiles(
            self.smiles[idx],
            self.token_to_id,
            self.max_len
        )

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        return x, y
