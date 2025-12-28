# verify_structure.py

import os

def check_directory(path, name):
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
        return True
    else:
        print(f"✗ {name}: {path} - NOT FOUND")
        return False

print("=" * 50)
print("PROJECT STRUCTURE VERIFICATION")
print("=" * 50)

base_dir = os.getcwd()
print(f"\nBase Directory: {base_dir}\n")

directories = [
    ("data/face_images", "Face Images Directory"),
    ("data/hand_images", "Hand Images Directory"),
    ("data/processed", "Processed Data Directory"),
    ("src", "Source Directory"),
    ("src/preprocessing", "Preprocessing Module"),
    ("src/feature_extraction", "Feature Extraction Module"),
    ("src/fuzzy_vault", "Fuzzy Vault Module"),
    ("src/blockchain", "Blockchain Module"),
    ("contracts", "Smart Contracts Directory"),
    ("models", "Models Directory"),
    ("tests", "Tests Directory"),
    ("config", "Config Directory"),
    ("logs", "Logs Directory"),
]

all_ok = True
for path, name in directories:
    if not check_directory(path, name):
        all_ok = False

print("\n" + "=" * 50)
if all_ok:
    print("All directories are set up correctly! ✓")
else:
    print("Some directories are missing. Please create them.")
print("=" * 50)