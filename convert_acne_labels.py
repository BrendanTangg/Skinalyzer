from pathlib import Path

# The three folders containing YOLO label files
label_folders = [
    Path("dataset/train/labels"),
    Path("dataset/valid/labels"),
    Path("dataset/test/labels")
]

for folder in label_folders:

    for label_file in folder.glob("*.txt"):

        # Read every bounding box in this label file
        lines = label_file.read_text().splitlines()

        new_lines = []

        for line in lines:
            parts = line.split()

            if len(parts) == 5:
                # YOLO format:
                # class x_center y_center width height
                #
                # Replace whatever the original class was
                # (0, 1, 2, or 3) with class 0 = acne
                parts[0] = "0"

                new_lines.append(" ".join(parts))

        # Replace the original label file
        label_file.write_text("\n".join(new_lines))

print("Finished converting all labels to class 0 = acne")