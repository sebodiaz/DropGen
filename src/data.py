import csv
import monai
import argparse
import glob
import augmentations as aug
import numpy as np
import os
import nibabel as nib
import monai.transforms as mt
import shutil
import re
import torch
from typing import List, Dict

# do not change
SHUFFLE_SEED = 123

# dataloader
DATASET_LOADERS = {}


def register_dataset(name):
    """Decorator to register dataset loading functions"""
    def decorator(func):
        DATASET_LOADERS[name] = func
        return func

    return decorator


def get_dataset(opts: argparse.Namespace):
    """Returns the dataset file paths for the specified dataset.

    If --split_csv is provided, uses it to assign subjects to splits from --data_dir.
    If --data_dir is provided, uses generic nnUNet-style loader.
    Otherwise, falls back to hardcoded dataset-specific loaders.
    """
    rng = np.random.default_rng(SHUFFLE_SEED)
    if getattr(opts, "split_csv", None) is not None:
        if getattr(opts, "data_dir", None) is None:
            raise ValueError("--split_csv requires --data_dir to be set.")
        return load_from_split_csv(opts)
    if getattr(opts, "data_dir", None) is not None:
        return load_nnunet_style(opts, rng)
    try:
        return DATASET_LOADERS[opts.dataset](opts, rng)
    except KeyError:
        raise ValueError(
            f"Unknown dataset {opts.dataset}. Available: {list(DATASET_LOADERS)}"
        )


def get_dataloaders(opts: argparse.Namespace):
    """Returns the dataloader for the specified dataset"""
    # get the file path names
    files = get_dataset(opts)  # dictionary of 'image' and 'label' paths
    
    # rng
    rng = np.random.default_rng(SHUFFLE_SEED)
    if opts.num_subjects is not None:
        rng.shuffle(files["train"])
        files["train"] = files["train"][: opts.num_subjects]

    # load the dataset using CacheDataset
    train_datasets = monai.data.CacheDataset(
        data=files["train"],
        transform=aug.get_train_transforms(opts),
        cache_rate=opts.cache_rate,
        num_workers=4,
    )
    # now load dataloader
    dataloader = monai.data.DataLoader(
        train_datasets,
        batch_size=opts.batch_size,
        sampler=torch.utils.data.RandomSampler(
            train_datasets,
            replacement=True,
            num_samples=opts.max_steps * opts.batch_size,
        ),
        num_workers=12,
        pin_memory=True,
        collate_fn=monai.data.list_data_collate,
        drop_last=True,  # for torch.compile()
        persistent_workers=True,
    )

    # validation dataloader
    val_dataset = monai.data.CacheDataset(
        data=files["val"],
        transform=aug.get_eval_transforms(opts),
        cache_rate=opts.cache_rate,
        num_workers=2,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=monai.data.list_data_collate,
        drop_last=False,
    )

    # test dataloader
    test_dataset = monai.data.CacheDataset(
        data=files["test"],
        transform=aug.get_eval_transforms(opts),
        cache_rate=0.0,
        num_workers=0,
    )

    test_loader = monai.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=monai.data.list_data_collate,
        drop_last=False,
    )

    print(
        f"Loaded {opts.dataset}: \n Train samples: {len(train_datasets)} \n Val samples: {len(val_dataset)} \n Test samples: {len(test_dataset)}"
    )

    return {"train": dataloader, "val": val_loader, "test": test_loader}



def load_from_split_csv(opts: argparse.Namespace):
    """Load dataset using a CSV file with image/label paths and split assignments.

    Expected CSV columns: 'image', 'label', 'split' (one of train, val, test).
    Subject is derived from the label filename.
    """

    splits = {"train": [], "val": [], "test": []}
    with open(opts.split_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_name = row["split"]
            subject = os.path.basename(row["label"]).replace(".nii.gz", "")
            if split_name not in splits:
                raise ValueError(f"Invalid split '{split_name}' for '{subject}'. Must be train, val, or test.")
            splits[split_name].append({
                "image": row["image"],
                "label": row["label"],
                "subject": subject,
            })

    # standardize (resample to target spacing)
    spacing_str = "x".join(str(s) for s in opts.spacing)
    for split in splits:
        if splits[split]:
            splits[split] = standardize_dataset(
                splits[split],
                os.path.join(opts.data_dir, f"{split}_resampled_{spacing_str}"),
                target_spacing=opts.spacing,
            )

    return splits


def load_nnunet_style(opts: argparse.Namespace, rng: np.random.Generator):
    """Generic loader for datasets organized in nnUNet-style directory structure.

    Expected layout under opts.data_dir:
        imagesTr/   labelsTr/       (training)
        imagesVal/  labelsVal/      (validation)
        imagesTs/   labelsTs/       (testing, optional)

    Image files use nnUNet channel suffix: <subject>_0000.nii.gz
    Label files match by subject name:    <subject>.nii.gz
    """
    data_dir = opts.data_dir

    split_dirs = {
        "train": ("imagesTr", "labelsTr"),
        "val": ("imagesVal", "labelsVal"),
        "test": ("imagesTs", "labelsTs"),
    }

    splits = {}
    for split, (img_folder, lab_folder) in split_dirs.items():
        img_dir = os.path.join(data_dir, img_folder)
        lab_dir = os.path.join(data_dir, lab_folder)

        if not os.path.isdir(img_dir):
            if split == "test":
                splits[split] = []
                continue
            raise FileNotFoundError(
                f"Expected directory {img_dir} for split '{split}'. "
                f"See README for the expected nnUNet-style folder layout."
            )

        images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        files = []
        for img_path in images:
            basename = os.path.basename(img_path)
            # strip nnUNet channel suffix: subject_0000.nii.gz -> subject.nii.gz
            label_name = re.sub(r"_\d{4}\.nii\.gz$", ".nii.gz", basename)
            label_path = os.path.join(lab_dir, label_name)
            if not os.path.exists(label_path):
                print(f"[WARN] Missing label for {basename}, expected {label_name}")
                continue
            subject = label_name.replace(".nii.gz", "")
            files.append({"image": img_path, "label": label_path, "subject": subject})

        splits[split] = files

    # standardize (resample to target spacing)
    spacing_str = "x".join(str(s) for s in opts.spacing)
    for split in splits:
        if splits[split]:
            splits[split] = standardize_dataset(
                splits[split],
                os.path.join(data_dir, f"{split}_resampled_{spacing_str}"),
                target_spacing=opts.spacing,
            )

    return splits


@register_dataset("brats")
def load_brats(opts: argparse.Namespace, rng: np.random.Generator):
    """Loads the BraTS dataset"""
    # images
    flair_images  = sorted(glob.glob('.../msd-brats/FLAIR/BRATS_*_FLAIR.nii.gz'))
    t2_images     = sorted(glob.glob('.../msd-brats/T2/BRATS_*_T2.nii.gz'))
    t1w_images    = sorted(glob.glob('.../msd-brats/T1w/BRATS_*_T1w.nii.gz'))
    t1wgad_images = sorted(glob.glob('.../msd-brats/T1wGAD/BRATS_*_T1wGAD.nii.gz'))
    
    # segmentations
    segs          = sorted(glob.glob('.../msd-brats/labs/BRATS_*.nii.gz'))

    # extract subject ids
    subject_ids  = []
    for img_path in flair_images:
        subject_id = img_path.split('/')[-1].split('_')[1]
        subject_ids.append(subject_id)

    # create mapping
    all_subjects = {}
    for idx, subject_id in enumerate(subject_ids):
        all_subjects[subject_id] = {
            "flair": flair_images[idx],
            "t2": t2_images[idx],
            "t1w": t1w_images[idx],
            "t1wgad": t1wgad_images[idx],
            "seg": segs[idx]
        }
    
    # get the list of unique subjects and shuffle them
    unique_subjects = list(all_subjects.keys())
    rng.shuffle(unique_subjects)
    
    # split subjects according to desired ratio
    train_subjects = unique_subjects[:340]    # 1020 = (340 * 3)
    val_subjects   = unique_subjects[340:388] # 48
    test_subjects  = unique_subjects[388:]    # 96
    
    # create data dictionaries for each split
    train_files    = []
    val_files      = []
    test_files     = []
    
    # populate training set
    for subject_id in train_subjects:
        subject_data = all_subjects[subject_id]
        # Add each modality as a separate training example, including the subject_id
        train_files.append({"image": subject_data["t2"], "label": subject_data["seg"], "subject": subject_id})
        train_files.append({"image": subject_data["t1w"], "label": subject_data["seg"], "subject": subject_id})
        train_files.append({"image": subject_data["t1wgad"], "label": subject_data["seg"], "subject": subject_id})

    # Populate validation set (using FLAIR)
    for subject_id in val_subjects:
        subject_data = all_subjects[subject_id]
        val_files.append({"image": subject_data["flair"], "label": subject_data["seg"], "subject": subject_id})
        
    # Populate test set (using FLAIR)
    for subject_id in test_subjects:
        subject_data = all_subjects[subject_id]
        test_files.append({"image": subject_data["flair"], "label": subject_data["seg"], "subject": subject_id})

    # standardize dataset (resample if needed)
    train_files = standardize_dataset(
        train_files,
        f".../msd-brats/TRresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )

    val_files = standardize_dataset(
        val_files,
        f".../msd-brats/VAresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    
    test_files = standardize_dataset(
        test_files,
        f".../msd-brats/TEresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    
    # create dictionary
    dataset_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    return dataset_dict


@register_dataset("hvsmr")
def load_hvsmr(opts: argparse.Namespace, rng: np.random.Generator):
    """Loads the HVSMR 2.0 dataset"""
    images = sorted(
        glob.glob(
            ".../hvsmr2.0/cropped/pat*_cropped.nii.gz"
        )
    )
    segs = sorted(
        glob.glob(".../hvsmr2.0/labs/pat*_labels.nii.gz")
    )

    # put all files into standard monai format
    all_files = [
        {"image": img, "label": lbl, "basename": os.path.basename(img).split("_")[0]}
        for img, lbl in zip(images, segs)
    ]

    # show train subjects
    train_HVSMR = [
        "pat0",
        "pat1",
        "pat2",
        "pat5",
        "pat6",
        "pat8",
        "pat10",
        "pat11",
        "pat13",
        "pat18",
        "pat26",
        "pat27",
        "pat7",
        "pat14",
        "pat15",
        "pat19",
        "pat23",
        "pat24",
        "pat25",
        "pat29",
        "pat48",
        "pat17",
        "pat12",
    ]

    # split into train and val based on subject IDs
    train_files = [f for f in all_files if f["basename"] in train_HVSMR]
    val_test_files = [f for f in all_files if f["basename"] not in train_HVSMR]

    # check if train_files length is the same as train_HVSMR
    assert len(train_files) == len(
        train_HVSMR
    ), "Train files length mismatch!. they are {} vs {}".format(
        len(train_files), len(train_HVSMR)
    )

    # shuffle and split
    rng.shuffle(val_test_files)
    split_idx = len(val_test_files) // 2
    val_files, test_files = val_test_files[:split_idx], val_test_files[split_idx:]

    # standardize dataset
    train_files = standardize_dataset(
        train_files,
        f".../hvsmr2.0/TRresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        opts.spacing,
    )

    val_files = standardize_dataset(
        val_files,
        f".../hvsmr2.0/VAresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        opts.spacing,
    )

    test_files = standardize_dataset(
        test_files,
        f".../hvsmr2.0/TEresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        opts.spacing,
    )
    
    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


@register_dataset("amos")
def load_amos(opts: argparse.Namespace, rng: np.random.Generator):
    """Loads the AMOS2022 dataset"""

    # file paths
    tr_images = (
        ".../amos22processed/train_resampled_1x1x1.5/img/"
    )
    tr_labels = ".../amos22processed/train_resampled_1x1x1.5/label/"
    va_images = ".../amos22processed/valtest/img/"
    va_labels = ".../amos22processed/valtest/label/"
    # get file lists
    train_images = sorted(glob.glob(tr_images + "*.nii.gz"))
    train_labels = sorted(glob.glob(tr_labels + "*.nii.gz"))
    val_images = sorted(glob.glob(va_images + "*.nii.gz"))
    val_labels = sorted(glob.glob(va_labels + "*.nii.gz"))

    # parse into standard monai dataset format
    train_files = [
        {"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)
    ]
    val_files = [
        {"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)
    ]

    # shuffle validation files and then split in half for val and test
    rng.shuffle(val_files)
    split_idx = len(val_files) // 2
    val_files, test_files = val_files[:split_idx], val_files[split_idx:]

    # standard dataset
    train_files = standardize_dataset(
        train_files,
        f".../amos22processed/TRresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )

    val_files = standardize_dataset(
        val_files,
        f".../amos22processed/VAresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )

    test_files = standardize_dataset(
        test_files,
        f".../amos22processed/TEresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

@register_dataset("cow")
def load_cow(opts: argparse.Namespace, rng: np.random.Generator):
    """Loads the TopCoW dataset"""
    
    # set data directories
    dirs = {
        "ct_img": ".../topcow2024/ct_img/",
        "mr_img": ".../topcow2024/mr_img/",
        "ct_lab": ".../topcow2024/ct_label_new/",
        "mr_lab": ".../topcow2024/mr_label_new/",
    }
    
    # TopCoW is named weirdly....
    extract_subject = lambda fn: int(re.search(r"_(\d{3})_", fn).group(1))
    
    def label_path(fn: str, lab_dir: str) -> str:
        base = re.sub(r'_\d{4}\.nii\.gz$', '', fn)
        return os.path.join(lab_dir, base + ".nii.gz")

    def load_split(img_dir: str, lab_dir: str) -> List[Dict]:
        """Load image-label pairs, skipping missing labels."""
        data = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.nii.gz"))):
            fn = os.path.basename(img_path)
            lab_path = label_path(fn, lab_dir)
            if not os.path.exists(lab_path):
                print(f"[WARN] Missing label for {fn}, expected {os.path.basename(lab_path)}")
                continue

            data.append({
                "image": img_path,
                "label": lab_path,
                "subject": extract_subject(fn),
            })
        return data
    
    # load the CT (train) and MR (val/test) splits
    train_files = load_split(dirs["ct_img"], dirs["ct_lab"])
    val_test_files = load_split(dirs["mr_img"], dirs["mr_lab"])
    
    # build subject splits
    subjects = sorted({d["subject"] for d in val_test_files})
    rng.shuffle(subjects)
    
    # split of 75 / 20 / rest
    train_subjects = set(subjects[:75])
    val_subjects = set(subjects[75:95])
    test_subjects = set(subjects[95:])
    
    train_data = [d for d in train_files   if d["subject"] in train_subjects]
    val_data   = [d for d in val_test_files if d["subject"] in val_subjects]
    test_data  = [d for d in val_test_files if d["subject"] in test_subjects]
    
    # standardize dataset
    train_data = standardize_dataset(
        train_data,
        f".../topcow2024/TRresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    val_data = standardize_dataset(
        val_data,
        f".../topcow2024/VAresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    test_data = standardize_dataset(
        test_data,
        f".../topcow2024/TEresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

@register_dataset("prostate")
def load_prostate(opts: argparse.Namespace, rng: np.random.Generator):
    """Loads the multi-site prostate dataset"""
    base_dir = ".../multi-site-prostate/raw/"
    img_dir = os.path.join(base_dir, "imgs")
    lab_dir = os.path.join(base_dir, "labs")
    
    train_sites = {"RUNMC", "I2CVB"}
    val_sites   = {"UCL", "HK"}
    test_sites  = {"BIDMC", "BMC"}
    
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(lab_dir, "*.nii.gz")))
    
    assert len(image_files) == len(label_files), \
        f"Number of images and labels do not match: {len(image_files)} and {len(label_files)}"
    
    train_files, val_files, test_files = [], [], []
    for img_path, lbl_path in zip(image_files, label_files):
        filename = os.path.basename(img_path)
        site     = filename.split("_")[0]
        pair     = {"image": img_path,
                    "label": lbl_path,
                    "subject": filename.replace(".nii.gz", "")}

        if site in train_sites:
            train_files.append(pair)
        elif site in val_sites:
            val_files.append(pair)
        elif site in test_sites:
            test_files.append(pair)
        else:
            print(f"[WARNING] Unknown site: {site} in file {filename}")
    
    # standardize dataset
    train_files = standardize_dataset(
        train_files,
        f".../prostate_multi_site/TRresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    val_files = standardize_dataset(
        val_files,
        f".../prostate_multi_site/VAresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    test_files = standardize_dataset(
        test_files,
        f".../prostate_multi_site/TEresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    
@register_dataset("pancreas")
def load_pancreas(opts: argparse.Namespace, rng: np.random.Generator):
    """ Load the pancreas dataset """
    train_images = sorted(glob.glob('.../PancreasDG/train/images/*.nii.gz'))
    train_labels = sorted(glob.glob('.../PancreasDG/train/labels/*.nii.gz'))
    
    ## val
    val_images   = sorted(glob.glob('.../PancreasDG/val/images/*.nii.gz'))
    val_labels   = sorted(glob.glob('.../PancreasDG/val/labels/*.nii.gz'))
    
    ## test
    test_images  = sorted(glob.glob('.../PancreasDG/test/images/*.nii.gz'))
    test_labels  = sorted(glob.glob('.../PancreasDG/test/labels/*.nii.gz'))
    
    # helper function to create file dicts
    def make_files(images, labels):
        files = []
        for img, lbl in zip(images, labels):
            subject_img = os.path.basename(img).replace(".nii.gz", "")
            files.append({
                "image": img,
                "label": lbl,
                "subject": subject_img
            })
        return files

    train_files = make_files(train_images, train_labels)
    val_files   = make_files(val_images, val_labels)
    test_files  = make_files(test_images, test_labels)
    
    # standardize dataset
    train_files = standardize_dataset(
        train_files,
        f".../PancreasDG/TRresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    val_files = standardize_dataset(
        val_files,
        f".../PancreasDG/VAresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    test_files = standardize_dataset(
        test_files,
        f".../PancreasDG/TEresampled_{opts.spacing[0]}x{opts.spacing[1]}x{opts.spacing[2]}".format(
            opts=opts
        ),
        target_spacing=opts.spacing,
    )
    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    
    

def check_spacing(path):
    """Returns the spacing of a NIfTI image given its file path"""
    img = nib.load(path)
    return img.header.get_zooms()


def needs_resampling(current_spacing, target_spacing):
    """Checks if the current spacing differs from the target spacing"""
    return not np.allclose(current_spacing, target_spacing, atol=1e-3)


def standardize_dataset(file_list: list, out_dir: str, target_spacing=(1.0, 1.0, 1.5)):
    """
    file_list: list of dicts with 'image' and 'label' keys
    out_dir: output directory to save resampled files
    target_spacing: desired spacing for resampling (in mm); default is (1.0, 1.0, 1.5) mm
    """
    # use MONAI's expected subfolder names so SaveImaged can create them
    out_img_dir = os.path.join(out_dir, "image")
    out_lab_dir = os.path.join(out_dir, "label")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)

    # check if resampling files already exist
    all_exist = all(
        os.path.exists(os.path.join(out_img_dir, os.path.basename(f["image"])))
        and os.path.exists(os.path.join(out_lab_dir, os.path.basename(f["label"])))
        for f in file_list
    )

    # count mismatched files
    mismatch_count = sum(
        needs_resampling(check_spacing(f["image"]), target_spacing) for f in file_list
    )

    if mismatch_count > 0 and not all_exist:
        # resample everything
        print(
            f"[{out_dir} {mismatch_count}] Resampling all files to spacing {target_spacing} mm..."
        )
        transform = mt.Compose(
            [
                mt.LoadImaged(keys=["image", "label"]),
                mt.EnsureChannelFirstd(keys=["image", "label"]),
                mt.Spacingd(
                    keys=["image", "label"],
                    pixdim=target_spacing,
                    mode=("bilinear", "nearest"),
                ),
                RemoveChannelFirstd(keys=["image", "label"]),
                mt.SaveImaged(
                    keys=["image"],
                    output_dir=out_img_dir,
                    output_postfix="",  # don't append an extra postfix (keep original extension)
                    separate_folder=False,
                ),
                mt.SaveImaged(
                    keys=["label"],
                    output_dir=out_lab_dir,
                    output_postfix="",
                    separate_folder=False,
                ),
            ]
        )
        ds = monai.data.Dataset(data=file_list, transform=transform)
        dl = monai.data.DataLoader(ds, batch_size=1, num_workers=4)
        for _ in dl:
            pass

    # no resampling is needed, but we need to move files
    elif not all_exist:
        # just move/copy files to the standardized folder
        print(f"[{out_dir}] All files OK; moving to standardized folder...")
        for f in file_list:
            shutil.copy(
                f["image"], os.path.join(out_img_dir, os.path.basename(f["image"]))
            )
            shutil.copy(
                f["label"], os.path.join(out_lab_dir, os.path.basename(f["label"]))
            )

    # no sampling or moving is needed
    else:
        print(f"[{out_dir}] All files already standardized.")

    # return updated file list
    new_files = [
        {
            "image": os.path.join(out_img_dir, os.path.basename(f["image"])),
            "label": os.path.join(out_lab_dir, os.path.basename(f["label"])),
        }
        for f in file_list
    ]
    return new_files


class RemoveChannelFirstd(mt.MapTransform):
    """Removes channel dimension from images in a dictionary"""

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                d[key] = img[0]
        return d
