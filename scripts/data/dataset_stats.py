import os

def count_images(path):
    return len([
        f for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

base = "data"
classes = ["real", "ai_generated", "ai_edited"]

print("Dataset Statistics")
print("------------------")

total = 0
for cls in classes:
    count = count_images(os.path.join(base, cls))
    total += count
    print(f"{cls}: {count} images")

print("------------------")
print(f"Total images: {total}")
