from collections import Counter
import matplotlib.pyplot as plt
import statistics

file = "smiles_train.txt"

lengths = []
characters = Counter()

with open(file, "r") as f:
    for line in f:
        smiles = line.strip()

        if not smiles:
            continue

        lengths.append(len(smiles))
        characters.update(smiles)


print(f"Number of molecules: {len(lengths)}")
print(f"Average length: {statistics.mean(lengths):.2f}")
print(f"Median length: {statistics.median(lengths)}")
print(f"Minimum length: {min(lengths)}")
print(f"Maximum length: {max(lengths)}")

lengths_sorted = sorted(lengths)
print(
    f"95th percentile length: {lengths_sorted[int(0.95 * len(lengths_sorted))]}")
print(
    f"99th percentile length: {lengths_sorted[int(0.99 * len(lengths_sorted))]}")

print("\nMost common characters:")
for character, count in characters.most_common(30):
    print(f"{repr(character):>5}: {count}")

print("\nVocabulary size:", len(characters))
print("Vocabulary:", sorted(characters.keys()))

plt.hist(lengths, bins=50)
plt.xlabel("SMILES length")
plt.ylabel("Count")
plt.title("Distribution of SMILES Lengths")
# plt.savefig("length_distribution.png")
plt.show()
