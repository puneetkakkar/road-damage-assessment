import os
import shutil
from utils.utils import check_if_directory_exists
from tqdm import tqdm


# Here in this function, we process raw images, and copy them
# into the processed folder
def process_raw_images(input_img_dir, output_img_dir):
    if not check_if_directory_exists(output_img_dir):
        os.makedirs(output_img_dir)

    # List all files in the input image directory
    files = os.listdir(input_img_dir)

    # Here, we iterate over each file in the directory and process it
    for filename in tqdm(files, desc="Processing image files", unit="file"):

        # Check if the file is an image file (with .jpg, .jpeg, or .png extensions)
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
        ):
            # Build the full path to the source and destination files
            source = os.path.join(input_img_dir, filename)
            destination = os.path.join(output_img_dir, filename)

            try:
                # Copy the image file from source to destination
                shutil.copy(source, destination)
            except Exception as e:
                print(f"Error processing file {source}: {e}")
