import os

# Folder containing images/ and meta/ (~ means your home folder)
DATA_DIR = os.path.expanduser("~/datasets/food-101")

CATEGORIES = [
    "hamburger", "hot_dog", "pizza", "tacos", "chicken_wings", "french_fries", "poutine",
    "pancakes", "waffles", "omelette",
    "chocolate_cake", "ice_cream", "donuts",
    "spaghetti_bolognese", "ramen", "pho", "fried_rice",
    "steak", "caesar_salad", "greek_salad",
]

def load_split(split):
    """Return a list of (image_path, label) for one split: 'train' or 'test'."""
    items = []
    with open(os.path.join(DATA_DIR, "meta", f"{split}.txt")) as f:
        for line in f:
            entry = line.strip()                 # e.g. "pizza/1001116"
            category = entry.split("/")[0]       # "pizza"
            if category in CATEGORIES:
                path = os.path.join(DATA_DIR, "images", entry + ".jpg")
                items.append((path, CATEGORIES.index(category)))
    return items

if __name__ == "__main__":
    train = load_split("train")
    test = load_split("test")

    # Check: should print 15000 and 5000
    print("Total:", len(train), "train,", len(test), "test")

    # Check: every category should have 750 train / 250 test
    for i, c in enumerate(CATEGORIES):
        n_tr = sum(1 for _, y in train if y == i)
        n_te = sum(1 for _, y in test if y == i)
        print(f"{c:22s} {n_tr} {n_te}")

    # Check: the image files actually exist
    missing = [p for p, _ in train + test if not os.path.exists(p)]
    print("Missing image files:", len(missing))