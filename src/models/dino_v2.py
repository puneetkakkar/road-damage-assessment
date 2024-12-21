import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


# Here, we define a linear classifier that applies a convolutional layer to the embeddings
class LinearClassifier(nn.Module):
    def __init__(self, in_channels, token_w=32, token_h=32, num_labels=1):
        super(LinearClassifier, self).__init__()
        self.in_channels = in_channels
        self.width = token_w
        self.height = token_h

        # The classifier is a 2D convolution with kernel size (1, 1), applied to each token
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        # Reshape the input embeddings to (batch_size, height, width, channels)
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)

        # Change the tensor shape to (batch_size, channels, height, width) for convolution
        embeddings = embeddings.permute(0, 3, 1, 2)

        # Apply the 1x1 convolution to get the final logits (predictions)
        return self.classifier(embeddings)


# This is the main model for road damage semantic segmentation using Dinov2
class Dinov2RoadDamageSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # The core model (Dinov2) for extracting feature embeddings
        self.dinov2 = Dinov2Model(config)

        # The classifier that maps the embeddings to the final segmentation logits
        self.classifier = LinearClassifier(config.hidden_size, 32, 32, 1)

    def forward(self, pixel_values, labels=None):
        # Get the outputs from the Dinov2 model including token embeddings
        outputs = self.dinov2(pixel_values)

        # Extract the patch embeddings, ignoring the first token which is the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # Pass the patch embeddings through the classifier to get the segmentation logits
        logits = self.classifier(patch_embeddings)

        # Upsample the logits to match the input image size using bilinear interpolation
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        # Compute the loss
        loss = None
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(1), labels.float())

        return SemanticSegmenterOutput(loss=loss, logits=logits)
