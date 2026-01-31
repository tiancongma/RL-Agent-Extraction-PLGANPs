from pathlib import Path

MAX_ITEMS = 30
OUTPUT_FILE = "project_structure.txt"

def write_tree(path: Path, file, prefix: str = ""):
    entries = sorted(
        path.iterdir(),
        key=lambda p: (p.is_file(), p.name.lower())
    )

    total = len(entries)
    shown = entries[:MAX_ITEMS]

    for i, entry in enumerate(shown):
        is_last = (i == len(shown) - 1) and (total <= MAX_ITEMS)
        connector = "└─ " if is_last else "├─ "
        file.write(prefix + connector + entry.name + "\n")

        if entry.is_dir():
            extension = "   " if is_last else "│  "
            write_tree(entry, file, prefix + extension)

    if total > MAX_ITEMS:
        omitted = total - MAX_ITEMS
        file.write(prefix + f"└─ ... ({omitted} more items omitted)\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    out_path = root / OUTPUT_FILE

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{root.name}/\n")
        write_tree(root, f)

    print(f"[OK] Project structure written to: {out_path}")
