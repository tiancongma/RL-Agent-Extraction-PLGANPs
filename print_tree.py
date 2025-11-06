from pathlib import Path

def print_tree(path: Path, prefix: str = ""):
    files = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, f in enumerate(files):
        connector = "└─ " if i == len(files) - 1 else "├─ "
        print(prefix + connector + f.name)
        if f.is_dir():
            extension = "   " if i == len(files) - 1 else "│  "
            print_tree(f, prefix + extension)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(f"{root.name}/")
    print_tree(root)
