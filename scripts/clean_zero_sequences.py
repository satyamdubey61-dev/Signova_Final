"""Find and delete all-zero .npy sequence files from Data/sequences/Hello/."""
import os, numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hello_dir = os.path.join(BASE_DIR, "Data", "sequences", "Hello")

deleted = []
kept = []
for f in sorted(os.listdir(hello_dir)):
    if not f.endswith('.npy'):
        continue
    path = os.path.join(hello_dir, f)
    data = np.load(path)
    if np.all(data == 0):
        os.remove(path)
        deleted.append(f)
        print(f"DELETED (all-zero): {f}")
    else:
        kept.append(f)

print(f"\nDeleted {len(deleted)} corrupted files: {deleted}")
print(f"Kept {len(kept)} valid files")
