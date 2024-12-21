import argparse
import os
import sys
import stat
from data_preprocessing.process_annotations import process_raw_annotations
from data_preprocessing.process_images import process_raw_images
from data_preprocessing.generate_binary_masks import process_images_and_annotations
from training.dino_v2.train import train as train_dino_v2
from training.segformer.train import train as train_segformer
from inference.infer_dino_v2 import inference as infer_dino_v2
from inference.infer_segformer import inference as infer_segformer
from utils.utils import (
    load_config,
    check_if_directory_exists,
    is_directory_empty,
    get_config_key,
    count_files,
)


# This function processes the dataset by checking if the data has already been processed
# or not. If not, it processes raw data (annotations, images) and generates binary masks
# after processing the annotations and images from the raw dataset
def process_dataset(config):
    raw_data_dir_path = get_config_key(config, "data.raw_data_dir")
    raw_data_img_dir_path = f"{raw_data_dir_path}/img"
    raw_data_ann_dir_path = f"{raw_data_dir_path}/ann"

    processed_dir_path = get_config_key(config, "data.processed_data_dir")
    processed_img_dir_path = f"{processed_dir_path}/img"
    processed_ann_dir_path = f"{processed_dir_path}/ann"
    processed_mask_dir_path = f"{processed_dir_path}/mask"

    if not check_if_directory_exists(processed_dir_path):
        os.makedirs(processed_dir_path)

    original_ann_file_count = 0
    original_img_file_count = 0
    if check_if_directory_exists(raw_data_ann_dir_path):
        original_ann_file_count = count_files(raw_data_ann_dir_path)
    else:
        print("Raw data's annotation directory not found!!")
        return

    print(f"Found {original_ann_file_count} raw annotations")

    if check_if_directory_exists(raw_data_img_dir_path):
        original_img_file_count = count_files(raw_data_img_dir_path)
    else:
        print("Raw data's image directory not found!!")
        return

    print(f"Found {original_img_file_count} raw annotations")

    processed_ann_file_count = 0
    processed_img_file_count = 0
    processed_mask_file_count = 0
    if check_if_directory_exists(processed_ann_dir_path):
        processed_ann_file_count = count_files(processed_ann_dir_path)

    if check_if_directory_exists(processed_img_dir_path):
        processed_img_file_count = count_files(processed_img_dir_path)

    if check_if_directory_exists(processed_mask_dir_path):
        processed_mask_file_count = count_files(processed_mask_dir_path)

    if (
        is_directory_empty(processed_dir_path)
        or (processed_ann_file_count < original_ann_file_count)
        or (processed_img_file_count < original_img_file_count)
    ):

        print("Processed dataset not found.")

        # Process the annotations if they are not found or found incomplete
        # in the processed folder
        if processed_ann_file_count < original_ann_file_count:

            print("Processing annotations...")

            input_ann_dir = raw_data_ann_dir_path
            output_ann_dir = processed_ann_dir_path

            # Processing raw dataset annotations to required format
            process_raw_annotations(input_ann_dir, output_ann_dir)

        # Process the images if they are not found or found incomplete
        # in the processed folder
        if processed_img_file_count < original_img_file_count:

            print("Processing images...")

            input_img_dir = raw_data_img_dir_path
            output_img_dir = processed_img_dir_path

            # Processing raw dataset images to required format
            process_raw_images(input_img_dir, output_img_dir)

        # Generate masks if they are not found or incomplete
        if processed_mask_file_count < original_ann_file_count:
            print("Generating masks...")

            process_images_and_annotations(
                processed_img_dir_path,
                processed_ann_dir_path,
                processed_mask_dir_path,
            )

        # change_permissions_recursively(processed_dir_path, 0o755)

        print("Data processing is complete. Proceeding with operation...")
    else:
        print("Processed dataset found. Proceeding with operation...")


def change_permissions_recursively(path, mode):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), mode)
        for file in files:
            os.chmod(os.path.join(root, file), mode)


# Here, we train selected models based on the input list of models
def train_models(config, models_to_train):
    if "dino_v2" in models_to_train:
        print("Training DINOv2 model...")
        train_dino_v2(config, "dino_v2")

    if "segformer" in models_to_train:
        print("Training SegFormer model...")
        train_segformer(config, "segformer")

    print("Model training complete!")


# This function helps to run inference with selected models
def run_inference_models(config, models_to_infer):
    if "dino_v2" in models_to_infer:
        print("Running inference with DINOv2 model...")
        infer_dino_v2(config, "dino_v2")

    if "segformer" in models_to_infer:
        print("Running inference with SegFormer model...")
        infer_segformer(config, "segformer")

    print("Inference complete!")


def main():
    parser = argparse.ArgumentParser(
        description="RADAR: Road Analysis and Damage Assessment Research"
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=["train", "inference", "both"],
        required=True,
        help="Operation to perform: 'train', 'inference', or 'both'",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["dino_v2", "segformer", "sam2.1"],
        required=True,
        help="List of models to train/infer: 'dino_v2', 'segformer', 'sam2.1'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Process the raw dataset and generate binary masks
    process_dataset(config)

    # If operation is 'train' or 'both', train the selected models
    if (args.operation == "train" or args.operation == "both") and args.models:
        train_models(config, args.models)

    # If operation is 'inference' or 'both', run inference on the selected models
    if args.operation == "inference" or args.operation == "both":
        run_inference_models(config, args.models)


if __name__ == "__main__":
    main()
