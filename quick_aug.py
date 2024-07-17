import os
import json
import imageio
from PIL import Image as PILImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm

def load_image_correct_orientation(image_path):
    image = PILImage.open(image_path)
    if image.format == 'MPO':
        # Check for EXIF orientation and correct it
        exif = image._getexif()
        orientation_key = 274  # cf ExifTags
        if exif and orientation_key in exif:
            orientation = exif[orientation_key]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    return np.array(image.convert('RGB'))

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def format_points(points):
    # Convert points to integers but keep them in float format for consistency
    return [[float(round(x)), float(round(y))] for x, y in points]

def clip_points_to_image(points, img_height, img_width):
    # Ensures that all points are within the image boundaries
    clipped_points = []
    for x, y in points:
        clipped_x = np.clip(x, 0, img_width - 1)
        clipped_y = np.clip(y, 0, img_height - 1)
        clipped_points.append([clipped_x, clipped_y])
    return clipped_points

def augment_image_and_labels(image, json_data, seq, image_height, image_width):

    # Prepare polygons from JSON data
    polygons = [Polygon(format_points(shape['points'])) for shape in json_data['shapes'] if shape['shape_type'] == 'polygon']
    polygons_on_image = PolygonsOnImage(polygons, shape=(image_height, image_width, image.shape[2]))

    # Apply augmentation
    image_aug, polygons_aug = seq(image=image, polygons=polygons_on_image)

    # Update JSON data for the augmented polygons
    new_json_data = json_data.copy()
    for shape, polygon_aug in zip(new_json_data['shapes'], polygons_aug.polygons):
        clipped_points = clip_points_to_image(polygon_aug.exterior.tolist(), image_height, image_width)
        shape['points'] = format_points(clipped_points)
    
    new_json_data['imageHeight'] = image_height
    new_json_data['imageWidth'] = image_width

    return image_aug, new_json_data

def main(source_folder, destination_folder, num_augmentations=10, h=480, w=640):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # flip horizontally
        iaa.Affine(
            rotate=(-45, 45),  # rotate between -30 and +30 degrees
            scale=(0.5, 1.2),  # scale between 80% and 120%
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        ),
        iaa.Multiply((0.4, 1.5)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 2.0)),  # blur
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
        iaa.ContrastNormalization(alpha=(0.5, 2.0)),
        iaa.Grayscale(alpha=(0.0, 0.5)),
        # iaa.Fog(),
        iaa.CropToAspectRatio(1.333333, position="uniform"),  # Crop to aspect ratio of 3:4 (0.75)
        iaa.Resize({"height": h, "width": w})  # Resize the image to 480x640
    ])

    # Use tqdm to create a progress bar
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for filename in tqdm(files, desc="Processing images"):  # tqdm wraps around any iterable
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            base_name = filename.split('.')[0]
            json_path = os.path.join(source_folder, base_name + '.json')
            image_path = os.path.join(source_folder, filename)

            if os.path.exists(json_path):
                # Load the original image and its JSON for each augmentation
                for i in range(num_augmentations):
                    json_data = load_json(json_path)
                    image = load_image_correct_orientation(image_path)
                    
                    aug_image, aug_json_data = augment_image_and_labels(image, json_data, seq, h, w)
                    aug_image_path = os.path.join(destination_folder, f"{base_name}_{i}.jpg")
                    imageio.imwrite(aug_image_path, aug_image)
                    aug_json_data['imagePath'] = os.path.basename(aug_image_path)
                    aug_json_data['imageHeight'] = json_data['imageHeight']
                    aug_json_data['imageWidth'] = json_data['imageWidth']
                    save_json(aug_json_data, os.path.join(destination_folder, f"{base_name}_{i}.json"))

if __name__ == "__main__":
    source_folder = '/home/bentengma/work_space/quick_augmentation/small_set_in'
    destination_folder = '/home/bentengma/work_space/quick_augmentation/small_set_out'
    main(source_folder, destination_folder, num_augmentations=100)
