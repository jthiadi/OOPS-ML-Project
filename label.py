import os
from pathlib import Path


def relabel_paper_to_five(base_dir: str = "Dataset/valid/ipad") -> None:
    """
    Change the class id (first number in each line) of all YOLO label txt files
    in train/valid/test under `base_dir` to 5.

    It will modify files in-place.
    """
    base_path = Path(base_dir)
    subsets = ["train", "valid", "test"]

    for subset in subsets:
        labels_dir = base_path
        if not labels_dir.is_dir():
            print(f"Skipping missing directory: {labels_dir}")
            continue

        txt_files = list(labels_dir.glob("*.txt"))
        print(f"Processing {len(txt_files)} files in {labels_dir}")

        for txt_path in txt_files:
            # Read all lines
            with txt_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            changed = False

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    # Keep empty lines as-is
                    new_lines.append(line)
                    continue

                parts = stripped.split()
                # Ensure we at least have a class id and some coords
                if len(parts) >= 1:
                    if parts[0] != "9":
                        parts[0] = "9"
                        changed = True

                    # Reconstruct the line with a single space separator and newline
                    new_line = " ".join(parts) + "\n"
                    new_lines.append(new_line)
                else:
                    # Unexpected format; keep original line
                    new_lines.append(line)

            if changed:
                with txt_path.open("w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"Updated: {txt_path}")
            else:
                # No class ids changed in this file
                pass


if __name__ == "__main__":
    relabel_paper_to_five()


