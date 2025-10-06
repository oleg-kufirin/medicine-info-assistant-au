import os
import shutil

count = 0
for root, dirs, files in os.walk(".."):
    for d in dirs:
        if d == "__pycache__":
            path = os.path.join(root, d)
            shutil.rmtree(path)
            print(f"Removed: {path}")
            count += 1

print(f"\n✅ Done. Removed {count} __pycache__ folder(s).")
