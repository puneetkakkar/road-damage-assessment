import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from utils.utils import get_config_key, select_device
from data.dataset import get_dataset
from data.dataset import get_dino_v2_transforms
from models.dino_v2 import Dinov2RoadDamageSemanticSegmentation


# This function loads an image, converts it to RGB, and applies transformations
# to prepare it for inference
def load_and_preprocess_image(image_path, input_size=448):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the transformation pipeline from the DINO V2 model
    _, transform = get_dino_v2_transforms()

    # Apply transformations to the image (e.g., resizing, normalization)
    transformed = transform(image=image)
    image = transformed["image"]

    # Convert the image into a PyTorch tensor and
    # change the channel order (from HWC to CHW)
    image = torch.tensor(image).permute(2, 0, 1).float()

    return image


# This function runs inference on either a custom image or a
# random test image from a dataset
def run_inference(
    model,
    dataset=None,
    custom_image_path=None,
    use_custom_image=False,
    input_size=448,
    output_path=None,
):
    # Select the device (GPU or CPU) to run the model on
    device = select_device()

    # If a custom image is provided, preprocess it
    if use_custom_image and custom_image_path is not None:
        image = load_and_preprocess_image(custom_image_path, input_size).to(device)
        original_image = Image.open(custom_image_path)
        original_image = np.array(original_image)
        mask = None

    # If a dataset is provided, select a random image and its corresponding mask
    elif not use_custom_image and dataset is not None:
        test_image_idx = np.random.randint(0, len(dataset))
        image, mask, original_image, original_mask = dataset[test_image_idx]
        mask = mask.to(device)
    else:
        raise ValueError("Either dataset or custom_image_path must be provided.")

    # Add a batch dimension to the image and send it to
    # the selected device (GPU or CPU)
    image = image.unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations since we are in inference mode
    with torch.no_grad():
        # Run the image through the model to get the output
        outputs = model(image)

    # Get the original image size to properly resize the output mask
    original_size = original_image.shape[:2]

    # Upsample the output logits to match the original image size
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )

    # Apply sigmoid to get probabilities, then threshold to create a binary mask
    pred = torch.sigmoid(upsampled_logits).cpu().numpy()
    pred_binary = (pred > 0.5).astype(np.uint8).squeeze()

    # Visualize and save the inference results
    visualize_inference(original_image, pred_binary, output_path, mask)


# This function visualizes the inference results by overlaying the
# predicted binary mask on the original imag
def visualize_inference(
    original_image, pred_binary, output_path, ground_truth_mask=None
):
    # Initialize a color image for the predicted segmentation (red color for damage areas)
    color_seg = np.zeros(
        (pred_binary.shape[0], pred_binary.shape[1], 3), dtype=np.uint8
    )

    # Set the red color for pixels where the binary mask is 1 (predicted damage)
    color_seg[pred_binary == 1] = [255, 0, 0]

    # Convert BGR to RGB for visualization
    color_seg = color_seg[..., ::-1]

    # Overlay the predicted segmentation on top of the original image
    img_and_mask = np.array(original_image) * 0.5 + color_seg * 0.5
    img_and_mask = img_and_mask.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_and_mask)
    plt.title(f"Predicted Mask Overlay", fontsize=16)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")


# This function handles the overall inference process, loading the model,
# running inference, and saving results
def inference(config, model_name):
    # Load the checkpoint file path from the configuration
    fine_tuned_model_checkpoint = get_config_key(
        config, f"models.{model_name}_checkpoint"
    )

    # Check if we should use a custom image for inference
    use_custom_image = get_config_key(
        config, f"inference.{model_name}.use_custom_image"
    )

    # Get the base model name for loading the pre-trained model
    base_model_name = get_config_key(config, f"training.{model_name}.base_model")

    # Get the path where the inference results should be saved
    inference_save_path = get_config_key(
        config, f"inference.{model_name}.inference_save_path"
    )

    device = select_device()

    # Load the model from the pre-trained checkpoint
    model = Dinov2RoadDamageSemanticSegmentation.from_pretrained(base_model_name).to(
        device
    )
    model.load_state_dict(torch.load(fine_tuned_model_checkpoint))
    model.to(device)

    # If we are using a custom image, run inference on it
    if use_custom_image:
        custom_image_path = get_config_key(
            config, f"inference.{model_name}.custom_image_path"
        )

        print("Running inference on a custom user image...")
        run_inference(
            model,
            custom_image_path=custom_image_path,
            use_custom_image=use_custom_image,
            output_path=inference_save_path,
        )

    # Otherwise, run inference on a random test image from the dataset
    else:
        _, _, test_dataset = get_dataset(config, model_name)

        print("Running inference on a random test image...")
        run_inference(
            model,
            dataset=test_dataset,
            use_custom_image=use_custom_image,
            output_path=inference_save_path,
        )
