import os
import zlib
import base64
import json
import cv2
import numpy as np

from utils.utils import check_if_directory_exists
from datetime import datetime
from tqdm import tqdm


# This function helps to decode a supervisely base64 bitmap (compressed) into a mask (image)
# Code ref: https://github.com/supervisely/docs/blob/master/data-organization/Annotation-JSON-format/04_Supervisely_Format_objects.md#bitmap
def decode_bitmap_to_mask(bitmap_data, origin):

    # Decode and decompress the bitmap data
    decoded_data = zlib.decompress(base64.b64decode(bitmap_data))

    # Convert the decoded data into a numpy array
    nparr = np.frombuffer(decoded_data, np.uint8)

    # Decode the numpy array into a grayscale image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Failed to decode the bitmap data.")

    return img


# It helps in converting the raw annotation data to a required structured format
def convert_annotation(annotation):

    image_metadata = annotation.get("size", {})

    image_height, image_width = image_metadata.get("height", 0), image_metadata.get(
        "width", 0
    )

    image_objects = annotation["objects"]

    image_date_captured = datetime.today()
    if len(image_objects) > 0:
        image_date_captured = image_objects[0]["createdAt"]

    image_metadata = {
        "height": image_height,
        "width": image_width,
        "date_captured": image_date_captured,
    }

    converted_annotation = {
        "image": image_metadata,
        "annotations": [],
    }

    segmentation_annotations = []

    for obj in annotation["objects"]:
        category_id = obj["classId"]

        bitmap_data = obj["bitmap"]["data"]
        origin = obj["bitmap"]["origin"]

        mask = decode_bitmap_to_mask(bitmap_data, origin)

        area = int(np.sum(mask > 0))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            annotation_obj = {
                "category_id": category_id,
                "segmentation": [],
                "area": area,
            }

            segmentation_points = []

            contour_points = contour.flatten().tolist()
            contour_points = [
                (x + origin[0], y + origin[1])
                for x, y in zip(contour_points[::2], contour_points[1::2])
            ]
            segmentation_points.extend([point for xy in contour_points for point in xy])
            annotation_obj["segmentation"] = [segmentation_points]

            if contours:
                x, y, w, h = cv2.boundingRect(contour)
                adjusted_bbox = [x + origin[0], y + origin[1], w, h]
                annotation_obj["bbox"] = adjusted_bbox

            segmentation_annotations.append(annotation_obj)

    converted_annotation["annotations"] = segmentation_annotations

    return converted_annotation


# This function parses and process all raw annotation files and saves the final
# required annotation json in the processed directory
def process_raw_annotations(input_ann_dir, output_ann_dir):

    # Check if the output directory exists, if not, create it
    if not check_if_directory_exists(output_ann_dir):
        os.makedirs(output_ann_dir)

    # List all files in the input annotation directory
    files = os.listdir(input_ann_dir)

    # Process each file in the directory
    for filename in tqdm(files, desc="Processing annotation files", unit="file"):

        # Only process .json files
        if filename.endswith(".json"):
            base_filename = os.path.splitext(filename)[0]

            # If the base filename ends with '.jpg', remove it (to match image file names)
            if base_filename.endswith(".jpg"):
                base_filename = base_filename[:-4]

            # Build the full paths for the input and output files
            input_file = os.path.join(input_ann_dir, filename)
            output_file = os.path.join(output_ann_dir, f"{base_filename}.json")

            # Open the raw annotation file and load its JSON data
            with open(input_file, "r") as f:
                annotation = json.load(f)

            # Convert the raw annotation to the format required by
            # DINOv2, SegFormer, and SAM2.1
            converted_annotation = convert_annotation(annotation)

            # Save the converted annotation to the output directory
            with open(output_file, "w") as f:
                json.dump(converted_annotation, f, indent=4)
