import torch
import wandb
import evaluate
import numpy as np
from torch import optim
from models.dino_v2 import Dinov2RoadDamageSemanticSegmentation
from data.dataset import get_data_loaders
from utils.utils import get_config_key, select_device
from datetime import datetime


# This is the main function for training the Dinov2 for road
# damage semantic segmentation model
def train(config, model_name):

    # Load training and validation data loaders
    train_loader, val_loader, _ = get_data_loaders(config, model_name)

    # Extract configuration parameters for training
    base_model_name = get_config_key(config, f"training.{model_name}.base_model")
    learning_rate = get_config_key(config, f"training.{model_name}.learning_rate")
    epochs = get_config_key(config, f"training.{model_name}.epochs")
    save_checkpoint_path = get_config_key(
        config, f"training.{model_name}.save_checkpoint_path"
    )
    wandb_project_name = get_config_key(config, "wandb.project_name")

    # Select the device (GPU/CPU)
    device = select_device()

    # Load the pre-trained Dinov2 model for road damage segmentation
    model = Dinov2RoadDamageSemanticSegmentation.from_pretrained(base_model_name)

    # Here, we are freezing the backbone of Dinov2 model to avoid training
    #  its parameters.
    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = False

    # Move the model to the selected device (GPU/CPU)
    model.to(device)

    # Defined the optimizer (AdamW) for training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialized the IoU (Intersection over Union) metric for evaluation
    iou_metric = evaluate.load("mean_iou")

    # We have commented the usage of wandb here because its usage requires
    # an API key for it to work. It is Used for logging the results on WandB platform.
    # wandb.init(project=wandb_project_name, name="DINOv2")

    best_val_iou = 0
    # Training loop: Iterate through the epochs
    for epoch in range(epochs):

        # Set model to training model
        model.train()
        total_train_loss = 0

        # Loop through the training data
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Forward pass (outputs logits and loss)
            outputs = model(images, labels=masks)

            # Extract the loss value
            loss = outputs.loss

            # Backward pass: Compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate the training loss
            total_train_loss += loss.item()

            # Used for logging the results on WandB platform
            # wandb.log({"train_loss": loss.item(), "epoch": epoch})

        # Validation phase after training the model for one epoch
        model.eval()
        total_val_loss = 0
        predictions, ground_truths = [], []

        # No need to compute gradients during evaluation
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images, labels=masks)
                total_val_loss += outputs.loss.item()

                # Apply sigmoid activation to logits to obtain binary predictions
                pred = torch.sigmoid(outputs.logits).cpu().numpy()
                pred_binary = (pred > 0.5).astype(
                    np.uint8
                )  # Threshold to get binary mask

                # Append predictions and ground truths for IoU calculation
                predictions.append(pred_binary.squeeze())
                ground_truths.append(masks.cpu().numpy())

        predictions = np.concatenate(predictions)
        ground_truths = np.concatenate(ground_truths)

        # Compute the validation IoU metric
        val_metrics = iou_metric.compute(
            predictions=predictions,
            references=ground_truths,
            num_labels=2,
            ignore_index=255,
        )

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {total_val_loss / len(val_loader):.4f}")
        print(f"Mean IoU: {val_metrics['mean_iou']}")

        # Save the model checkpoint if validation IoU improves
        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            torch.save(
                model.state_dict(),
                f"{save_checkpoint_path}/fine_tuned_dinov2_{datetime.today()}.pt",
            )

        # Used for logging the results on WandB platform
        # wandb.log(
        #     {
        #         "val_loss": total_val_loss / len(val_loader),
        #         "mean_iou": val_metrics["mean_iou"],
        #         "epoch": epoch,
        #     }
        # )
