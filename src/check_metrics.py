"""
Evaluate a generated molecule submission.

Computes:
- Validity
- Uniqueness
- Novelty

using RDKit and the training dataset.
"""

from rdkit import Chem

TRAIN_FILE = "smiles_train.txt"
SUBMISSION = "submissions/submission_final.txt"

print("Loading files...")
with open(TRAIN_FILE) as f:
    train_set = set(line.strip() for line in f if line.strip())

with open(SUBMISSION) as f:
    submission = [line.strip() for line in f if line.strip()]

if len(submission) == 0:
    raise ValueError("Submission file is empty.")

# Validity
valid = []
for smi in submission:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid.append(Chem.MolToSmiles(mol))  # canonicalize

# Uniqueness
unique = set(valid)

# Novelty
novel = unique - train_set

print(f"Total submitted : {len(submission)}")
print(f"Validity        : {len(valid)/len(submission):.3f}")
print(f"Uniqueness      : {len(unique)/len(submission):.3f}")
print(f"Novelty         : {len(novel)/len(submission):.3f}")
