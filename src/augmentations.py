import monai.transforms as mt
import numpy as np
import argparse

""" File defines augmentation pipeline for training and, validation, and testing """


def get_train_transforms(opts: argparse.Namespace):

    if opts.dataset in ["amos", "cow"]:
        intensity_norm = mt.ScaleIntensityRanged(
            keys=["image"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
        )
        orientation_augs = [mt.Identityd(keys=["image", "label"])]
    else:
        intensity_norm = mt.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        orientation_augs = [
            mt.RandFlipd(keys=["image", "label"], prob=opts.aug, spatial_axis=0),
            mt.RandAxisFlipd(keys=["image", "label"], prob=opts.aug),
            mt.RandRotate90d(
                keys=["image", "label"], prob=opts.aug, max_k=3, spatial_axes=(0, 1)
            ),
        ]


    return mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.EnsureTyped(keys=["image", "label"]),
            # Spatial normalization
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=opts.spacing,
                mode=("bilinear", "nearest"),
            ),
            mt.Orientationd(
                keys=["image", "label"], axcodes="RAS"
            ),  # optional but recommended
            mt.SpatialPadd(keys=["image", "label"], spatial_size=opts.crop_size),
            # Intensity normalization (only once)
            intensity_norm,
            # Crop before heavy augmentations
            mt.RandSpatialCropd(
                keys=["image", "label"], roi_size=opts.crop_size, random_size=False
            ),
            # Orientation flips + rotations
            *orientation_augs,
            # Image-only degradations
            mt.RandSimulateLowResolutiond(
                keys=["image"], prob=opts.aug, zoom_range=(0.25, 1.0)
            ),
            mt.RandGaussianNoised(keys=["image"], prob=opts.aug),
            mt.RandBiasFieldd(keys=["image"], prob=opts.aug, coeff_range=(0.0, 0.1)),
            mt.RandGibbsNoised(keys=["image"], prob=opts.aug, alpha=(0.0, 0.33)),
            mt.RandAdjustContrastd(keys=["image"], prob=opts.aug),
            mt.RandGaussianSmoothd(
                keys=["image"],
                prob=opts.aug,
                sigma_x=(0.0, 0.1),
                sigma_y=(0.0, 0.1),
                sigma_z=(0.0, 0.1),
            ),
            mt.RandGaussianSharpend(keys=["image"], prob=opts.aug),
            # Geometric transforms must be last among augs
            mt.RandAffined(
                keys=["image", "label"],
                prob=opts.aug,
                rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                shear_range=(0.2, 0.2, 0.2),
                scale_range=(0.2, 0.2, 0.2),
                spatial_size=opts.crop_size,
                padding_mode="zeros",
                mode=("bilinear", "nearest"),
            ),
            # final intensity
            mt.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            
            # Final tensor conversion
            mt.ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )


def get_eval_transforms(opts: argparse.Namespace):
    """Returns the evaluation augmentation pipeline"""    

    eval_transforms = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.EnsureTyped(keys=["image", "label"]),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=opts.spacing,
                mode=("bilinear", "nearest"),
            ), # not needed since we resample offline, but keeping for consistency / safety
            mt.Orientationd(
                keys=["image", "label"], axcodes="RAS"
            ),  # for the sake of consistency
            mt.ScaleIntensityd(keys=["image"]),
            
            mt.ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )
    return eval_transforms


if __name__ == "__main__":
    """ Unit test for augmentation pipelines """
    import os
    from monai.data import DataLoader, Dataset

    # define options
    class Opts:
        def __init__(self):
            self.spacing = [1.0, 1.0, 1.0]
            self.crop_size = [128, 128, 64]
            self.aug = 0.5
            self.dataset = "amos"

    opts = Opts()

    # create dummy data
    data_dicts = [
        {
            "image": os.path.join("path", "to", "image1.nii.gz"),
            "label": os.path.join("path", "to", "label1.nii.gz"),
        },
        {
            "image": os.path.join("path", "to", "image2.nii.gz"),
            "label": os.path.join("path", "to", "label2.nii.gz"),
        },
    ]

    # create dataset and dataloader
    train_transforms = get_train_transforms(opts)
    train_ds = Dataset(data=data_dicts, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    # iterate through the dataloader
    for batch_data in train_loader:
        images, labels = batch_data["image"], batch_data["label"]
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break  # just one batch for testing