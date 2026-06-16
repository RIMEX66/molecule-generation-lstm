"""
Create the final submission file.

The script filters generated SMILES by:
1. Validity (RDKit parsing)
2. Uniqueness
3. Novelty with respect to the training set

The first 10,000 valid, unique, and novel molecules are
written to the final submission file.
"""

from rdkit import Chem


TRAIN_FILE = "smiles_train.txt"
GENERATED_FILE = "submissions/generated_raw.txt"
OUTPUT_FILE = "submissions/submission_final.txt"

TARGET_SIZE = 10000

print("Loading training set...")

with open(TRAIN_FILE) as f:
    train_set = set(line.strip() for line in f if line.strip())

print("Filtering molecules...")

final_smiles = []
seen = set()

with open(GENERATED_FILE) as f:
    for line in f:
        smi = line.strip()

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # canonicalize
        smi = Chem.MolToSmiles(mol)

        # unique
        if smi in seen:
            continue

        # novel
        if smi in train_set:
            continue

        seen.add(smi)
        final_smiles.append(smi)

        if len(final_smiles) >= TARGET_SIZE:
            break

print(f"Collected {len(final_smiles)} molecules")

with open(OUTPUT_FILE, "w") as f:
    for smi in final_smiles:
        f.write(smi + "\n")

print(f"Saved to {OUTPUT_FILE}")
