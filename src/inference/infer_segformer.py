import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
from PIL import Image
from torchvision import transforms
from utils.utils import get_config_key, select_device
from data.dataset import get_dataset
from data.dataset import get_segformer_transforms
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


# This function loads an image, applies necessary transformations, and
# processes it using the Segformer model's processor
def load_and_preprocess_image(image_path, image_processor):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the segmentation transformation pipeline for the image
    _, transform = get_segformer_transforms()

    # Apply the transformation (e.g., resizing, normalization)
    transformed = transform(image=image)

    # Process the image using the model's image processor to prepare the input
    inputs = image_processor(transformed["image"], return_tensors="pt")

    # Squeeze the batch dimension for inference
    for k, v in inputs.items():
        inputs[k] = inputs[k].squeeze_()

    return inputs


# This function runs the inference on either a custom image or a
# random test image from the dataset
def run_inference(
    model,
    dataset=None,
    custom_image_path=None,
    use_custom_image=False,
    input_size=448,
    image_processor=None,
    output_path=None,
):
    device = select_device()

    # Check if custom image is provided, load and preprocess it
    if use_custom_image and custom_image_path is not None:
        image = load_and_preprocess_image(custom_image_path, image_processor).to(device)

        # Load the original image to visualize later
        original_image = Image.open(custom_image_path)
        original_image = np.array(original_image)
        mask = None

    # Else, use a random image from the dataset
    elif not use_custom_image and dataset is not None:
        test_image_idx = np.random.randint(0, len(dataset))
        image, original_image, _ = dataset[test_image_idx]
        # original_image = Image.open(test_image_paths[test_image_idx])
        # mask = mask.to(device)
        mask = None

    else:
        raise ValueError("Either dataset or custom_image_path must be provided.")

    # Get pixel values for the model and send them to the device (GPU/CPU)
    pixel_values = image["pixel_values"].unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference with no gradient tracking
    with torch.no_grad():

        # Get the model's output (logits)
        outputs = model(pixel_values=pixel_values)
        # logits = outputs.logits

    # Post-process the outputs to get the segmentation map
    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[original_image.shape[:2]]
    )[0]

    # Convert the result to a numpy array for visualization
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

    visualize_inference(original_image, predicted_segmentation_map, output_path, mask)


# This function visualizes the predicted segmentation mask by overlaying it on the original image
def visualize_inference(
    original_image, predicted_segmentation_map, output_path, ground_truth_mask=None
):
    # Create an empty color image for the segmentation mask
    color_seg = np.zeros(
        (predicted_segmentation_map.shape[0], predicted_segmentation_map.shape[1], 3),
        dtype=np.uint8,
    )

    palette = np.array(
        [[0, 0, 0], [255, 0, 0]]  # class 0 (background)
    )  # class 1 (foreground)

    # Apply the colors based on predicted segmentation map
    for label, color in enumerate(palette):
        color_seg[predicted_segmentation_map == label, :] = color

    # Convert from BGR to RGB for visualization
    color_seg = color_seg[..., ::-1]

    # Overlay the predicted mask on top of the original image
    img_and_mask = np.array(original_image) * 0.5 + color_seg * 0.5
    img_and_mask = img_and_mask.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_and_mask)
    plt.title(f"Predicted Mask Overlay", fontsize=16)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")


# This function handles the entire inference process, including
# model loading, inference, and saving results
def inference(config, model_name):

    # Load the model checkpoint file path from the config
    fine_tuned_model_checkpoint = get_config_key(
        config, f"models.{model_name}_checkpoint"
    )

    # Check if we should use a custom image for inference
    use_custom_image = get_config_key(
        config, f"inference.{model_name}.use_custom_image"
    )

    # Load the base model name from the configuration
    base_model_name = get_config_key(config, f"training.{model_name}.base_model")

    # Get the path to save the inference results
    inference_save_path = get_config_key(
        config, f"inference.{model_name}.inference_save_path"
    )

    # Initialize the image processor for the Segformer model
    image_processor = AutoImageProcessor.from_pretrained(base_model_name)

    device = select_device()

    # Load the Segformer model for semantic segmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Load the model weights from the fine-tuned checkpoint
    model.load_state_dict(torch.load(fine_tuned_model_checkpoint))
    model.to(device)

    # If using a custom image, run inference on it
    if use_custom_image:
        custom_image_path = get_config_key(
            config, f"inference.{model_name}.custom_image_path"
        )

        print("Running inference on a custom user image...")
        run_inference(
            model,
            custom_image_path=custom_image_path,
            use_custom_image=use_custom_image,
            image_processor=image_processor,
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
            image_processor=image_processor,
            output_path=inference_save_path,
        )
