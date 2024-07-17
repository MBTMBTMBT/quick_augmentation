import sys
import os
import json
from shutil import rmtree
from os.path import join, exists

def main():
    print(sys.argv)
    if len(sys.argv) != 4:
        print("Usage: python convert.py <dataset> <segment|bbox> <output_folder>")
        sys.exit(1)

    dataset = sys.argv[1]
    data_type = sys.argv[2]
    output_folder = sys.argv[3]
    assert data_type in ["segment", "bbox"], "Data type must be 'segment' or 'bbox'"

    src_dir = dataset
    label_dir = output_folder

    if exists(label_dir):
        rmtree(label_dir)
    os.makedirs(label_dir)

    files = [f for f in os.listdir(join(src_dir, "images")) if f.endswith('.json')]
    labels = set()

    for fn in files:
        with open(join(src_dir, "images", fn), 'r') as file:
            data = json.load(file)
            for shape in data['shapes']:
                labels.add(shape['label'])

    ids = {label: index for index, label in enumerate(labels)}

    for fn in files:
        with open(join(src_dir, "images", fn), 'r') as file:
            data = json.load(file)

        bb = [
            f"{ids[shape['label']]} " + 
            " ".join(f"{x/data['imageWidth']} {y/data['imageHeight']}" for x, y in shape['points'])
            for shape in data['shapes'] if shape['shape_type'] == "rectangle"
        ]

        seg = [
            f"{ids[shape['label']]} " + 
            " ".join(f"{x/data['imageWidth']} {y/data['imageHeight']}" for x, y in shape['points'])
            for shape in data['shapes'] if shape['shape_type'] == "polygon"
        ]

        if bb and data_type == "bbox":
            with open(join(label_dir, fn[:-5] + ".txt"), 'w') as file:
                file.write("\n".join(bb))

        if seg and data_type == "segment":
            with open(join(label_dir, fn[:-5] + ".txt"), 'w') as file:
                file.write("\n".join(seg))

    with open(join(output_folder, "config.yml"), 'w') as file:
        file.write(f"path: {src_dir}\ntrain: {join(src_dir, 'images')}\nval: {join(src_dir, 'images')}\ntest:\n")
        file.write("\nnames:\n" + "\n".join(f"  {index}: {label}" for index, label in enumerate(labels)))

if __name__ == "__main__":
    main()
