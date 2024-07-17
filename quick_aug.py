import os
import json
import shutil
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def augment_image_and_labels(image, json_data, num_augmentations=10):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # flip horizontally
        iaa.Affine(
            rotate=(-30, 30),  # rotate between -30 and +30 degrees
            scale=(0.8, 1.2)  # scale between 80% and 120%
        ),
        iaa.Multiply((0.8, 1.2)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # blur
    ])

    images_aug = [image] + [seq(image=image) for _ in range(num_augmentations)]
    return images_aug

def main(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            base_name = filename.split('.')[0]
            json_path = os.path.join(source_folder, base_name + '.json')
            image_path = os.path.join(source_folder, filename)

            if os.path.exists(json_path):
                image = Image.open(image_path)
                json_data = load_json(json_path)

                # Convert PIL image to numpy array for imgaug
                image_np = np.array(image)
                augmented_images = augment_image_and_labels(image_np, json_data)

                # Save original and augmented images and jsons
                for i, aug_image in enumerate(augmented_images):
                    aug_image_pil = Image.fromarray(aug_image)
                    aug_image_path = os.path.join(destination_folder, f"{base_name}_{i}.jpg")
                    aug_image_pil.save(aug_image_path)

                    # Here, you should also modify json_data accordingly
                    # Currently, it saves the same json for all images
                    aug_json_path = os.path.join(destination_folder, f"{base_name}_{i}.json")
                    save_json(json_data, aug_json_path)

if __name__ == "__main__":
    source_folder = 'path_to_your_input_folder'
    destination_folder = 'path_to_your_output_folder'
    main(source_folder, destination_folder)
