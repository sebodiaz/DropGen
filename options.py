import argparse
import monai
import os
import torch

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="3D/2D U-Net for medical image segmentation"
        )
        self._add_all_arguments()

    def _add_all_arguments(self):
        """Add all arguments to parser at initialization"""
        self._add_architecture_args()
        self._add_dataset_args()
        self._add_training_args()
        self._add_optimizer_args()
        self._add_device_args()
        self._add_augmentation_args()
        self._add_class_mapping_args()
        self._add_misc_args()
        self._add_wandb_args() if wandb_available else None
        self._resume_args()
        self._set_save_dir()
        self._set_torch_backend_options()
        self._anatomix_options()

    def _add_misc_args(self):
        """Add miscellaneous arguments"""
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed")

    def _add_architecture_args(self):
        """Add model architecture related arguments"""
        self.parser.add_argument("--dimension", type=int, default=3, choices=[2, 3])
        self.parser.add_argument("--in_channels", type=int, default=1)
        self.parser.add_argument("--out_channels", type=int, default=2, help="Number of output segmentation classes; this is usually set later based on the dataset.")
        self.parser.add_argument("--base_filters", type=int, default=20)
        self.parser.add_argument("--num_levels", type=int, default=5)

    def _add_dataset_args(self):
        self.parser.add_argument(
            "--dataset", type=str, default="brats", choices=[
                                                             "brats",
                                                             "amos",
                                                             "hvsmr",
                                                             "cow",
                                                             "prostate",
                                                             "pancreas"
                                                             ]
        )
        self.parser.add_argument(
            "--data_dir", type=str, default=None,
            help="Path to dataset in nnUNet-style format (imagesTr/, labelsTr/, imagesVal/, labelsVal/, imagesTs/, labelsTs/). "
                 "When provided, this overrides the hardcoded dataset paths."
        )
        self.parser.add_argument("--batch_size", type=int, default=4)
        self.parser.add_argument(
            "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Target spacing for resampling; this is usually set later based on the dataset."
        )
        self.parser.add_argument("--num_classes", type=int, default=4)
        self.parser.add_argument("--cache_rate", type=float, default=1.0)

    def _add_training_args(self):
        self.parser.add_argument(
            "--crop_size", type=int, nargs=3, default=[128, 128, 128]
        )
        self.parser.add_argument("--max_steps", type=int, default=250_000)
        self.parser.add_argument("--eval_interval", type=int, default=1_000)
        self.parser.add_argument(
            "--method", type=str, default="erm", choices=[
                                                          "erm",
                                                          "dropgen",   # `Ours`
                                                          ]
        )
        self.parser.add_argument("--dropout_prob", type=float, default=0.75)
        self.parser.add_argument(
            "--num_subjects",
            type=int,
            default=None,
            help="If set, use only N training subjects (for few-shot experiments).",
        )
        self.parser.add_argument(
            "--split_csv",
            type=str,
            default=None,
            help="Path to a CSV file defining train/val/test splits. "
                 "Expected columns: 'subject' and 'split' (one of train, val, test).",
        )

    def _add_optimizer_args(self):
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            choices=["cosine", "step", "none"],
            help="Learning rate scheduler type, this currently only support cosine annealing, but there are probably better ways.",
        )
        self.parser.add_argument("--lr", type=float, default=2e-4)

    def _add_device_args(self):
        self.parser.add_argument("--device", type=str, default="cuda:0")

    def _add_augmentation_args(self):
        self.parser.add_argument("--aug", type=float, default=0.33, help="Probability of applying each augmentation")

    def _add_class_mapping_args(self):
        self.parser.add_argument("--class_mapping", type=str, default=None, help="Class mapping dictionary as a string; e.g., '{\"background\":0, \"tumor\":1}'")

    def _add_wandb_args(self):
        self.parser.add_argument("--wandb_project", type=str, default="stable_reps")
        self.parser.add_argument("--wandb_entity", type=str, default=None)
        self.parser.add_argument("--run_name", type=str, default=None, help="Name of the run for logging purposes")

    def _resume_args(self):
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume training from latest checkpoint",
        )

    def _set_save_dir(self):
        self.parser.add_argument(
            "--save_dir",
            type=str,
            default=".../runs",
            help="Directory to save models and logs",
        )

    def _set_seed(self, seed: int):
        monai.utils.set_determinism(seed=seed)

    def _create_output_dir(self, args):
        output_dir = os.path.join(
            args.save_dir, f"{args.run_name}" if args.run_name else "default_run"
        )
        os.makedirs(output_dir, exist_ok=True)
        args.output_dir = output_dir

    def _set_torch_backend_options(self):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _apply_method_overrides(self, args):
        if args.method == "dropgen":
            args.in_channels = 17
        elif args.method in ["gin", "gin+ipa"]:
            args.in_channels = 3
        return args

    def _anatomix_options(self):
        self.parser.add_argument("--layer_index", type=int, default=59)
        self.parser.add_argument("--feature_dim", type=int, default=16)
        self.parser.add_argument(
            "--norm_type", type=str, default="batch"
        )
        pass

    def _apply_dataset_overrides(self, args):
        if args.dataset == "brats":
            args.num_classes = 4
            args.spacing = [1.0, 1.0, 1.0]
            args.crop_size = [128, 128, 128]
            args.out_channels = args.num_classes
            args.class_mapping = {
                "background": 0,
                "edema": 1,
                "non-enhancing tumor": 2,
                "enhancing tumor": 3,
            }
            args.cache_rate = 1.0
        elif args.dataset == "amos":
            args.num_classes = 16
            args.spacing = [1.0, 1.0, 1.5]
            args.crop_size = [192, 192, 128]
            args.out_channels = args.num_classes
            args.class_mapping = {
                "background": 0,
                "spleen": 1,
                "right kidney": 2,
                "left kidney": 3,
                "gallbladder": 4,
                "esophagus": 5,
                "liver": 6,
                "stomach": 7,
                "aorta": 8,
                "post cava": 9,
                "pancreas": 10,
                "right adrenal gland": 11,
                "left adrenal gland": 12,
                "duodenum": 13,
                "bladder": 14,
                "prostate/uterus": 15,
            }
            args.cache_rate = 1.0
        elif args.dataset == "hvsmr":
            args.num_classes = 9
            args.spacing = [0.77, 0.72, 0.72]
            args.crop_size = [128, 160, 128]
            args.out_channels = args.num_classes
            args.class_mapping = {
                "background": 0,
                "left ventricle": 1,
                "right ventricle": 2,
                "left atrium": 3,
                "right atrium": 4,
                "aorta": 5,
                "pulmonary artery": 6,
                "superior vena cava": 7,
                "inferior vena cava": 8,
            }
            args.cache_rate = 1.0
        
        elif args.dataset == "cow":
            args.num_classes = 14
            args.spacing = [0.3, 0.3, 0.6]
            args.crop_size = [192, 160, 128]
            args.out_channels = args.num_classes
            args.class_mapping = {
                "background": 0,
                "basilar artery": 1,
                "right PCA": 2,
                "left PCA": 3,
                "right ICA": 4,
                "right MCA": 5,
                "left ICA": 6,
                "left MCA": 7,
                "right PcomA": 8,
                "left PcomA": 9,
                "Acom": 10,
                "right ACA": 11,
                "left ACA": 12,
                "3rd A2": 13,
            }
            args.cache_rate = 1.0
        
        elif args.dataset == "prostate":
            args.num_classes = 2
            args.spacing = [0.51, 0.52, 0.125]
            args.crop_size = [256, 256, 64]
            args.out_channels = args.num_classes
            args.class_mapping = {
                "background": 0,
                "prostate": 1,
            }
            args.cache_rate = 1.0

        elif args.dataset == "pancreas":
            args.num_classes = 2
            args.spacing = [1.09, 1.09, 4.4]
            args.crop_size = [256, 256, 64]
            args.out_channels = args.num_classes
            args.class_mapping = {
                "background": 0,
                "pancreas": 1,
            }
            args.cache_rate = 1.0
        
        return args

    def _load_checkpoint(self, args):
        latest_checkpoint = os.path.join(args.output_dir, "latest.pth")
        if os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint, map_location=args.device)
            print(f"Resuming training from {latest_checkpoint} at step {checkpoint.get('step', 0)}")
            return checkpoint
        else:
            print(f"No checkpoint found at {latest_checkpoint}, starting fresh.")
            return None

    def _wandb_init(self, args, run_id=None):
        if wandb_available:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=vars(args),
                id=run_id,
                resume="allow" if run_id else None,
                group=args.dataset,
            )
            args.run_id = wandb.run.id

    def _parse_run_name(self, args):
        
        #
        # run names are usually [method]_[dataset]_[optional tags]_seed[seed]
        # for example a DropGen model with dataset AMOS, channel dropout 75, and batch normalization
        # might look like "dropgen_amos_cd75_bn_seed1234"
        #
        
        if args.run_name:
            parts = args.run_name.split("_")
            if len(parts) >= 2:
                args.method = parts[0]
                args.dataset = parts[1]

            # parse optional tags // most are for the `Stable Representations Enable Generalization in Medical Image Segmentation` paper
            if "cd" in args.run_name:
                cd_part = [p for p in parts if p.startswith("cd")]
                if cd_part:
                    try:
                        cd_value = int(cd_part[0].replace("cd", ""))
                        args.dropout_prob = cd_value / 100.0
                    except ValueError:
                        pass
            
            # extract seed if present // usually not applicable but useful for reproducibility
            if "seed" in args.run_name:
                seed_part = [p for p in parts if p.startswith("seed")]
                if seed_part:
                    try:
                        args.seed = int(seed_part[0].replace("seed", ""))
                    except ValueError:
                        pass
        return args

    def parse(self):
        """Parse args and optionally resume checkpoint/W&B"""
        args = self.parser.parse_args()
        args = self._apply_dataset_overrides(args)
        args = self._apply_method_overrides(args)
        self._set_seed(args.seed)
        self._create_output_dir(args)
        args = self._parse_run_name(args)

        run_id = None
        if getattr(args, "resume", False):
            checkpoint = self._load_checkpoint(args)
            if checkpoint:
                ck_args = checkpoint.get("args", {}) or {}
                for k, v in ck_args.items():
                    setattr(args, k, v)
                    # then set args.resume = True again
                args.resume = True
                    
                run_id = checkpoint.get("wandb_run_id", None)
                print(f"Resuming with wandb_run_id: {run_id}")

        if wandb_available and args.run_name != "tmp":
            self._wandb_init(args, run_id=run_id)
        
        # temporary override for output dir if run_name is "tmp"
        args.output_dir = f".../runs/{args.run_name}"
        args.save_dir = f".../runs/{args.run_name}"
        
        # print arguments
        print("\n" + "=" * 30)
        print("Parsed Arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        print("=" * 30 + "\n")

        
        
        return args


if __name__ == "__main__":
    opts = Options()
    args = opts.parse()
    print(args)