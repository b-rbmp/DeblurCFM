import numpy as np
import torch
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stops the training if the monitored metric doesn't improve (according to metric_direction)
    after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        metric_direction="minimize",
        initial_best_metric=None,
        save_best_only=True,
        save_optimizer=True,
        save_scheduler=None,
    ):
        """
        Args:
            patience (int): How many epochs to wait after the last time the monitored metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                            Default: print
            metric_direction (str): Direction in which to optimize the metric.
                                    Possible values: 'minimize' or 'maximize'.
                                    Default: 'minimize'
            initial_best_metric (float): Starting best metric value.
                                         If None, will be set to np.Inf for 'minimize', or -np.Inf for 'maximize'.
                                         Default: None
            save_best_only (bool): If True, only saves the best model. If False, saves every improvement.
                                   Default: True
            save_optimizer (bool): If True, saves optimizer state. If False, only saves model state.
                                   Default: True
            save_scheduler (torch.optim.lr_scheduler._LRScheduler): Optional scheduler to save state.
                                                                   Default: None
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric_direction = metric_direction.lower()
        self.save_best_only = save_best_only
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        if self.metric_direction not in ["minimize", "maximize"]:
            raise ValueError("metric_direction must be either 'minimize' or 'maximize'")

        # Determine the initial "best" metric
        if initial_best_metric is not None:
            self.best_metric = initial_best_metric
        else:
            self.best_metric = (
                np.Inf if self.metric_direction == "minimize" else -np.Inf
            )

        self.best_epoch = None
        self.early_stop = False
        self.val_losses = []  # Track validation losses for plotting

    def __call__(self, metric, model, optimizer, epoch, global_step):
        """
        Checks if there is an improvement in the monitored metric and updates
        early stopping parameters accordingly.

        Args:
            metric (float): Current value of the monitored metric.
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer used.
            epoch (int): Current epoch number.
            global_step (int): Current global step.
        """
        improvement = False
        previous_best_metric = self.best_metric
        self.val_losses.append(metric)  # Track validation losses

        if self.metric_direction == "minimize":
            # Improvement means the current metric is lower than the best metric by at least delta
            if metric < self.best_metric - self.delta:
                improvement = True
        else:
            # 'maximize' direction
            # Improvement means the current metric is higher than the best metric by at least delta
            if metric > self.best_metric + self.delta:
                improvement = True

        if improvement:
            self.best_metric = metric
            self.best_epoch = epoch
            if self.save_best_only:
                self.save_checkpoint(
                    metric, model, optimizer, global_step, previous_best_metric
                )
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"Early Stopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func("Early stopping triggered")

    def save_checkpoint(
        self, metric, model, optimizer, global_step, previous_best_metric=None
    ):
        """Saves model when metric improves."""
        if self.verbose:
            self.trace_func(
                f"Metric improved from {previous_best_metric:.6f} to {metric:.6f}. Saving model"
            )

        # Prepare checkpoint dictionary
        checkpoint = {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "best_metric": metric,
        }

        # Add optimizer state if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Add scheduler state if provided
        if self.save_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.save_scheduler.state_dict()

        # Add model-specific attributes
        for attr in ["std", "mean", "max_value", "min_value"]:
            if hasattr(model, attr):
                checkpoint[attr] = getattr(model, attr)

        # Save checkpoint
        torch.save(checkpoint, self.path)
