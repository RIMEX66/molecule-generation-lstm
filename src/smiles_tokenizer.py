import re

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

# Handles special SMILES tokens like [NH4+], Cl, Br and %10
SMILES_PATTERN = re.compile(r"(\[[^\[\]]+\]|Br|Cl|%\d{2}|.)")


def tokenize_smiles(smiles):
    return SMILES_PATTERN.findall(smiles)


def build_vocabulary(smiles_list):
    tokens = set()

    for smiles in smiles_list:
        tokens.update(tokenize_smiles(smiles))

    vocabulary = SPECIAL_TOKENS + sorted(tokens)

    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    return vocabulary, token_to_id, id_to_token


def encode_smiles(smiles, token_to_id, max_len=120):
    tokens = ["<bos>"] + tokenize_smiles(smiles) + ["<eos>"]
    ids = [token_to_id[token] for token in tokens]

    if len(ids) < max_len:
        padding = [token_to_id["<pad>"]] * (max_len - len(ids))
        ids = ids + padding
    else:
        ids = ids[:max_len]

    return ids


def decode_ids(ids, id_to_token):
    tokens = []

    for idx in ids:
        token = id_to_token[int(idx)]

        if token == "<eos>":
            break

        if token != "<pad>" and token != "<bos>":
            tokens.append(token)

    return "".join(tokens)


if __name__ == "__main__":
    examples = [
        "CCO",
        "CCCl",
        "CCBr",
    ]

    for smiles in examples:
        print(smiles, "-", tokenize_smiles(smiles))
