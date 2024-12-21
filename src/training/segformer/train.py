import torch
import wandb
import evaluate
import torch.optim as optim
import numpy as np
import torch.nn as nn
from data.dataset import get_data_loaders
from transformers import SegformerForSemanticSegmentation
from utils.utils import select_device, get_config_key
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime


# This is the main function for training the SegFormer for road
# damage semantic segmentation model
def train(config, model_name):

    # Load data loaders for training and validation
    train_loader, val_loader, _ = get_data_loaders(config, model_name)

    # Extract hyperparameters and configuration values from the config file
    base_model_name = get_config_key(config, f"training.{model_name}.base_model")
    learning_rate = get_config_key(config, f"training.{model_name}.learning_rate")
    epochs = get_config_key(config, f"training.{model_name}.epochs")
    save_checkpoint_path = get_config_key(
        config, f"training.{model_name}.save_checkpoint_path"
    )
    wandb_project_name = get_config_key(config, "wandb.project_name")

    # Select the device (GPU or CPU)
    device = select_device()

    # Initialize the model from a pre-trained Segformer checkpoint
    # Here, we have taken num_labels=2 because of binary
    # classification (semantic segmentation)
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Load the Mean IoU evaluation metric from the `evaluate` library
    iou_metric = evaluate.load("mean_iou")

    # Initialize a GradScaler for mixed precision training to speed up training and save memory
    scaler = GradScaler()

    # We have commented the usage of wandb here because its usage requires
    # an API key for it to work. It is Used for logging the results on WandB platform.
    # wandb.init(project=wandb_project_name, name="Segformer")

    best_val_iou = 0
    # Training loop: Iterate through the epochs
    for epoch in range(epochs):

        # Set model to training model
        model.train()
        total_train_loss = 0

        # Loop through the training data
        for step, batch in enumerate(train_loader):

            # Move data to the device (GPU/CPU)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Zero the gradients to prepare for the backward pass
            optimizer.zero_grad()

            # Here, we are using autocast to enable mixed precision training
            with autocast():
                # Perform a forward pass to compute the loss
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            # Backpropagate the gradients using scaled loss for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate training loss
            total_train_loss += loss.item()

            # Used for logging the results on WandB platform
            # wandb.log({"train_loss": loss.item(), "epoch": epoch})

        # Validation loop
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_references = []

        # No need to compute gradients during evaluation
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # Perform forward pass during evaluation
                outputs = model(pixel_values=pixel_values, labels=labels)
                total_val_loss += outputs.loss.item()

                logits = outputs.logits

                # Upsample the logits to the same resolution as the labels
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )

                # Convert logits to predicted labels (binary segmentation)
                predicted = upsampled_logits.argmax(dim=1)

                all_predictions.append(predicted.cpu().numpy())
                all_references.append(labels.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_references = np.concatenate(all_references)

        # Compute evaluation metrics (Mean IoU)
        val_metrics = iou_metric.compute(
            predictions=all_predictions,
            references=all_references,
            num_labels=2,
            ignore_index=255,
        )

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {total_train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {total_val_loss/len(val_loader):.4f}")
        print(f"Mean IoU: {val_metrics['mean_iou']}")

        # Save the model checkpoint if the validation IoU improves
        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            torch.save(
                model.state_dict(),
                f"{save_checkpoint_path}/fine_tuned_segformer_{datetime.today()}.pt",
            )

        # Used for logging the results on WandB platform
        # wandb.log(
        #     {
        #         "val_loss": total_val_loss / len(val_loader),
        #         "mean_iou": val_metrics["mean_iou"],
        #         "epoch": epoch,
        #     }
        # )
