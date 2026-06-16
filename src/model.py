import torch
import torch.nn as nn


class SmilesLSTM(nn.Module):
    """
    Token-level LSTM language model for molecular SMILES strings.

    Given an input sequence of SMILES tokens, the model predicts
    the next token at each position in the sequence.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.output(out)
        return logits, hidden
