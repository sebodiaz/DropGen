import torch
import monai
import argparse


class Loss(torch.nn.Module):
    """Custom loss function class that selects and computes the appropriate loss based on the specified method."""
    def __init__(self,
                 opts: argparse.Namespace):
        super(Loss, self).__init__()
        
        # store opts
        self.opts = opts
    
        # define a mapping from match to loss functions
        self.loss_map = {
            'erm': monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True),
            'dropgen': monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True),
        }

        if self.opts.method not in self.loss_map:
            raise ValueError(f"Loss function for method '{self.opts.method}' is not defined.")

        self.loss_fn = self.loss_map[self.opts.method]

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(predictions, targets)
        return {'loss': loss} if isinstance(loss, torch.Tensor) else loss