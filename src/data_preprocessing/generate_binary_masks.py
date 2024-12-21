import os
import cv2
import json
import numpy as np

from tqdm import tqdm

ROAD_CATEGORY_ID = 2963


# This function helps to load images
def load_image(image_file_path):
    return cv2.imread(image_file_path)


# It helps in loading the annotation JSON file.
def load_annotation(annotation_file_path):
    with open(annotation_file_path, "r") as annotation_file:
        return json.load(annotation_file)


# This function saves the binary mask to the desired location
def save_binary_mask(binary_mask, binary_mask_path):
    cv2.imwrite(binary_mask_path, binary_mask)


# This function helps to create a binary mask from a segmentation list
def create_mask_from_segmentation(image_shape, segmentation):
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Iterate through each polygon in the segmentation data and fill it in the mask
    for polygon in segmentation:

        # Convert the polygon list to a numpy array and reshape to (n, 2) for coordinates
        polygon = np.array(polygon).reshape(-1, 2)

        # Fill the polygon in the mask with white (255)
        cv2.fillPoly(mask, [polygon], color=(255, 255, 255))
    return mask


# It helps to generate a binary mask for an image given class annotations
def generate_binary_mask(image_height, image_width, class_annotations):
    binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # bboxes = []

    # Iterate through each annotation in the class annotations list
    for annotation in class_annotations:

        # Get the segmentation and category ID from the annotation
        segmentation = annotation.get("segmentation", [])
        category_id = annotation.get("category_id", None)

        # Skip annotations belonging to the road category (category ID defined earlier)
        if category_id == ROAD_CATEGORY_ID:
            continue

        # Generate a mask from the segmentation and combine it with the existing binary mask
        mask = create_mask_from_segmentation((image_height, image_width), segmentation)
        binary_mask = np.maximum(binary_mask, mask)

        # if annotation["bbox"]:
        #     bboxes.append(annotation["bbox"])

    return binary_mask


def process_images_and_annotations(
    image_dir_path, annotation_dir_path, masks_output_path
):
    # Ensure the output directory for masks exists, create it if not
    os.makedirs(masks_output_path, exist_ok=True)

    # Get a list of image file paths (only considering .jpg files)
    image_file_paths = [
        image_file
        for image_file in os.listdir(image_dir_path)
        if image_file.endswith(".jpg")
    ]

    # Loop through each image file and process it
    for image_file in tqdm(
        image_file_paths, desc="Generating Binary Masks", unit="file"
    ):

        # Build the full file paths for image and corresponding annotation file
        image_file_path = os.path.join(image_dir_path, image_file)
        annotation_file_path = os.path.join(
            annotation_dir_path, image_file.replace(".jpg", ".json")
        )

        # If the annotation file doesn't exist, remove the image and skip processing
        if not os.path.exists(annotation_file_path):
            os.remove(image_file_path)
            print(
                f"Annotation file not found for {image_file_path}. Removing image, and skipping..."
            )
            continue

        # Load the image and its corresponding annotation
        image = load_image(image_file_path)
        annotation = load_annotation(annotation_file_path)

        image_height, image_width = image.shape[:2]

        class_annotations = annotation.get("annotations", [])
        binary_mask = generate_binary_mask(image_height, image_width, class_annotations)

        # Create the output file name for the binary mask
        # (same name as the image but with .jpg extension)
        binary_mask_filename = (
            os.path.splitext(os.path.basename(image_file_path))[0] + ".jpg"
        )
        mask_output_path = os.path.join(masks_output_path, binary_mask_filename)

        save_binary_mask(binary_mask, mask_output_path)
