"""Script zum Ausgeben der Projektstruktur in eine Textdatei (Tree-Style)."""

from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
START = Path(__file__).resolve().parent
ROOT = START
OUT_FILE = START / "project_structure.txt"

# --------------------------------------------------
# Config
# --------------------------------------------------
EXCLUDE = {
    "__pycache__",
    ".git",
    "wandb",
    ".mypy_cache",
    ".vscode",
    "logs",
    "mlenv",
}

DATA_DIR_NAMES = {"data"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def write_line(fh, prefix, is_last, name):
    connector = "└── " if is_last else "├── "
    fh.write(prefix + connector + name + "\n")


# --------------------------------------------------
# Walker
# --------------------------------------------------
def walk(path: Path, prefix: str, fh):
    entries = [p for p in path.iterdir() if p.name not in EXCLUDE]

    dirs = sorted(p for p in entries if p.is_dir())
    files = sorted(p for p in entries if p.is_file())

    all_entries = dirs + files

    for i, p in enumerate(all_entries):
        is_last = i == len(all_entries) - 1
        write_line(fh, prefix, is_last, p.name)

        if p.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            if p.name in DATA_DIR_NAMES:
                walk_data(p, new_prefix, fh)
            else:
                walk(p, new_prefix, fh)

            # 🔹 Vertikale Trennlinie zwischen Geschwistern
            if not is_last:
                fh.write(prefix + "│\n")


def walk_data(path: Path, prefix: str, fh):
    entries = list(path.iterdir())

    dirs = sorted(p for p in entries if p.is_dir())
    files = sorted(p for p in entries if p.is_file())

    all_entries = dirs + files[:2]

    for i, p in enumerate(all_entries):
        is_last = i == len(all_entries) - 1
        write_line(fh, prefix, is_last, p.name)

        if p.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            walk_data(p, new_prefix, fh)

            # 🔹 Vertikale Trennlinie zwischen Geschwistern
            if not is_last:
                fh.write(prefix + "│\n")


# --------------------------------------------------
# Run
# --------------------------------------------------
with OUT_FILE.open("w", encoding="utf-8") as fh:
    fh.write(f"{ROOT.name}/\n")
    walk(ROOT, "", fh)

print(f"Projektstruktur geschrieben nach: {OUT_FILE}")
