import albumentations as A


# In this function, we initialize the data augmentations
# transformations for both training, validation, and test dataset
# For DINOv2
def get_dino_v2_transforms():

    # These are the mean and standard deviation for normalization which
    # could be used for DINOv2
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformation initialized for training data
    train_transform = A.Compose(
        [A.Resize(448, 448), A.HorizontalFlip(p=0.5), A.Normalize(mean=mean, std=std)],
        is_check_shapes=False,
    )

    # Transformation initialized for validation data
    val_transform = A.Compose(
        [A.Resize(448, 448), A.Normalize(mean=mean, std=std)],
        is_check_shapes=False,
    )

    return train_transform, val_transform


# In this function, we initialize the data augmentations
# transformations for both training, validation, and test dataset
# for SegFormer
def get_segformer_transforms():

    # Initializing the transformations for train dataset
    train_transform = A.Compose(
        [
            A.Resize(width=448, height=448),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ],
        is_check_shapes=False,
    )

    # Initializing the transformations for val dataset
    val_transform = A.Compose([A.Resize(width=448, height=448)], is_check_shapes=False)

    return train_transform, val_transform
