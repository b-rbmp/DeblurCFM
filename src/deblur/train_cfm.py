import sys
import os
import shutil

sys.path.append(os.getcwd())  # Add the current working directory to the path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import argparse
import namegenerator
import datetime
import wandb
import copy

from src.deblur.cfm.cfm import (
    ConditionalFlowMatcher,
)
from torchdiffeq import odeint_adjoint as odeint
from src.models.unet import UNetModel
from src.deblur.datasets.deblur import make_deblur_splits
from src.utils.early_stopping import EarlyStopping
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


# - Argument Parsing -
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Conditional Flow Matching for Image Deblurring"
    )
    # - Data/Paths -
    parser.add_argument(
        "--blurred_dir",
        type=str,
        default="./data/output_text_deblur/blurred",
        help="Directory containing blurred images",
    )
    parser.add_argument(
        "--sharp_dir",
        type=str,
        default="./data/output_text_deblur/sharp",
        help="Directory containing sharp images",
    )
    parser.add_argument("--img_height", type=int, default=272, help="Image height")
    parser.add_argument("--img_width", type=int, default=480, help="Image width")
    parser.add_argument(
        "--run_string", type=str, default="deblur_cfm", help="Run String"
    )
    parser.add_argument(
        "--enable_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable W&B logging",
    )

    # - Training -
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation sampling batch size (defaults to training batch_size)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--grad_clip_val",
        type=float,
        default=1.0,
        help="Gradient Clip Value (set to 0 or None to disable)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing an optimizer step",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mixed-precision",
    )

    # - Scheduler -
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="plateau",
        help="Type of learning rate scheduler [plateau, cosine, none]",
    )
    parser.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.5,
        help="Factor to multiply learning rate by when plateauing",
    )
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=5,
        help="Number of epochs to wait before reducing learning rate",
    )
    parser.add_argument(
        "--lr_plateau_min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for plateau scheduler",
    )
    parser.add_argument(
        "--lr_plateau_threshold",
        type=float,
        default=1e-4,
        help="Threshold for measuring the new optimum",
    )

    # - CFM Specific -
    parser.add_argument(
        "--flow_matching_method",
        type=str,
        default="vanilla",
        help="CFM type [vanilla, exact, target, variance_preserving, schrodinger_bridge]",
    )
    parser.add_argument(
        "--cfm_sigma",
        type=float,
        default=0.01,
        help="CFM noise level sigma (often small, e.g., 0.01 or 0.1)",
    )

    # - ODE Solver Specific (for validation sampling) -
    parser.add_argument(
        "--ode_solver",
        type=str,
        default="rk4",
        help="ODE solver for sampling (e.g., 'rk4', 'dopri5', 'euler')",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=10,
        help="Number of steps for ODE solver during sampling",
    )
    parser.add_argument(
        "--ode_atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for adaptive ODE solvers",
    )
    parser.add_argument(
        "--ode_rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for adaptive ODE solvers",
    )

    # - Model Specific Args (UNet) -
    parser.add_argument(
        "--num_channels", type=int, default=128, help="Base channel count for UNet"
    )
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="Number of residual blocks per level",
    )
    parser.add_argument(
        "--channel_mult", type=str, default="1,2,2,2", help="Channel multipliers"
    )
    parser.add_argument(
        "--attention_resolutions", type=str, default="8,16", help="Attention resolutions"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")


    # - Validation Sampling -
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=20,
        help="Number of images to generate during validation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu'). Auto-detects if None.",
    )

    # - EMA Model Saving -
    parser.add_argument(
        "--ema_model_saving",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable EMA model saving",
    )
    parser.add_argument(
        "--ema_model_saving_decay",
        type=float,
        default=0.9999,
        help="EMA model saving decay",
    )

    # - Early Stopping -
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--early_stopping_verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verbose early stopping",
    )
    parser.add_argument(
        "--early_stopping_delta",
        type=float,
        default=0.0,
        help="Delta for early stopping",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "psnr", "ssim"],
        help="Metric to use for early stopping",
    )

    # - Partial Metrics Evaluation -
    parser.add_argument(
        "--eval_partial_metrics_per_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate partial metrics at the end of each epoch (slow)",
    )

    return parser.parse_args()


# - Helper to save validation images -
def save_val_samples(epoch, blurred_cond, sharp_gt, generated, sample_dir):
    epoch_dir = os.path.join(sample_dir, f"epoch_{epoch+1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    blurred_cond, sharp_gt, generated = map(
        lambda x: x.cpu().detach(), [blurred_cond, sharp_gt, generated]
    )
    # Clamp and scale images from [-1, 1] to [0, 1] for saving
    blurred_cond, sharp_gt, generated = map(
        lambda x: (x.clamp(-1, 1) + 1) / 2.0, [blurred_cond, sharp_gt, generated]
    )

    num_samples = generated.shape[0]
    for i in range(num_samples):
        comparison_grid = torch.cat([blurred_cond[i], generated[i], sharp_gt[i]], dim=2)
        vutils.save_image(
            comparison_grid, os.path.join(epoch_dir, f"val_sample_{i:02d}.png")
        )
    print(f"Saved {num_samples} validation samples to {epoch_dir}")


# - ODE Dynamics Wrapper -
class FlowDynamics(nn.Module):
    def __init__(
        self, model: UNetModel, condition: torch.Tensor
    ):
        super().__init__()
        self.model = model
        self.condition = condition

    def forward(self, t, x):
        batch_size = x.shape[0]
        t_input = t.expand(batch_size) # Use expand instead of repeat for scalar t

        # Prepare spatial input
        current_condition = (
            self.condition
        )  # Assuming condition batch matches x batch size
        if self.condition.shape[0] != batch_size:
            print(
                f"Warning: ODE batch size {batch_size} mismatch condition {self.condition.shape[0]}. Trying basic slicing/repeat."
            )
            if (
                batch_size == 1 and self.condition.shape[0] > 1
            ):
                current_condition = self.condition[0:1]
            elif (
                self.condition.shape[0] == 1 and batch_size > 1
            ):  # Condition needs repeating
                current_condition = self.condition.expand(batch_size, -1, -1, -1)
            else:  # More complex mismatch
                raise RuntimeError(
                    f"Cannot easily reconcile ODE batch size {batch_size} and condition batch size {self.condition.shape[0]}"
                )
        model_input = torch.cat([x, current_condition], dim=1)
        vector_field = self.model(x=model_input, t=t_input)
        return vector_field


def ema(model, ema_model, decay):
    """Update the EMA model weights using exponential moving average."""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# - Main Training Function -
def main():
    args = parse_args()
    DEVICE = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {DEVICE}")

    val_batch_size = (
        args.val_batch_size if args.val_batch_size is not None else args.batch_size
    )
    print(f"Using validation sampling batch size: {val_batch_size}")

    # - Setup Directories -
    FOLDER = "src/deblur"  # Consider making this an argument
    RUN_ID = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + args.run_string
        + "_"
        + namegenerator.gen()
    )
    artifacts_base_folder = os.path.join(FOLDER, "artifacts", args.run_string)
    run_artifacts_folder = os.path.join(artifacts_base_folder, RUN_ID)
    checkpoints_folder = os.path.join(run_artifacts_folder, "checkpoints")
    samples_folder = os.path.join(run_artifacts_folder, "samples")
    os.makedirs(run_artifacts_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(samples_folder, exist_ok=True)
    print(f"Run artifacts will be saved in: {run_artifacts_folder}")

    # - Dataset -
    IMG_DIMS = (args.img_height, args.img_width)
    C = 3
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_DIMS),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )
    print(
        f"Loading dataset from: Blurred='{args.blurred_dir}', Sharp='{args.sharp_dir}'"
    )
    try:
        train_split, val_split = make_deblur_splits(
            args.blurred_dir, args.sharp_dir, transform, val_ratio=0.2
        )
    except Exception as e:
        print(f"Error creating dataset splits: {e}")
        sys.exit(1)
    train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    print(f"Train samples: {len(train_split)}, Val samples: {len(val_split)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # - Get Fixed Validation Batch for Sampling/Metrics -
    fixed_val_blurred, fixed_val_sharp, actual_val_samples = None, None, 0
    if len(val_split) > 0:
        num_to_collect = min(
            args.val_num_samples, len(val_split)
        )
        if num_to_collect > 0:
            collected_blurred, collected_sharp = [], []
            # Use val_batch_size for efficient loading of samples
            temp_val_loader = DataLoader(
                val_split, batch_size=val_batch_size, shuffle=False
            )
            total_collected = 0
            for batch in temp_val_loader:
                blurred, sharp = batch
                needed = num_to_collect - total_collected
                count = min(needed, blurred.shape[0])
                collected_blurred.append(blurred[:count])
                collected_sharp.append(sharp[:count])
                total_collected += count
                if total_collected >= num_to_collect:
                    break

            if collected_blurred:
                fixed_val_blurred = torch.cat(collected_blurred, dim=0).to(DEVICE)
                fixed_val_sharp = torch.cat(collected_sharp, dim=0).to(DEVICE)
                actual_val_samples = fixed_val_blurred.shape[0]
                if actual_val_samples < args.val_num_samples:
                    print(
                        f"Warning: Collected only {actual_val_samples} samples (requested {args.val_num_samples}, available {len(val_split)})."
                    )
                print(
                    f"Using {actual_val_samples} samples from validation set for visualization and metrics."
                )
            else:
                print("Warning: Could not retrieve samples from validation loader.")
        else:
            print("Warning: Requested 0 validation samples.")
    else:
        print("Warning: Validation set is empty. Skipping sampling and metrics.")
        args.eval_partial_metrics_per_epoch = False  # Disable metrics if no val data

    # - Model -
    model_input_channels = 6
    print(f"Configuring UNet with input channels = {model_input_channels}")
    channel_mult = tuple(map(int, args.channel_mult.split(",")))
    try:
        if args.attention_resolutions.strip():
            attention_resolutions_list = [
                int(ds_factor.strip()) for ds_factor in args.attention_resolutions.split(',') if ds_factor.strip()
            ]
            for ds_factor in attention_resolutions_list:
                if ds_factor <= 0 or (ds_factor > 1 and (ds_factor & (ds_factor - 1)) != 0 and ds_factor != 1) :
                    print(f"Warning: Attention resolution ds_factor {ds_factor} is not a power of 2. This might be unintended for standard UNet ds values.")
        else:
            attention_resolutions_list = [] # No attention if empty string

        attention_resolutions_tuple = tuple(attention_resolutions_list)
        print(
            f"Using attention at downsample factors (ds): {attention_resolutions_tuple}"
        )
    except ValueError as e:
        print(
            f"Error: Invalid attention resolutions format '{args.attention_resolutions}'. Must be comma-separated integers (downsample factors). Error: {e}"
        )
        sys.exit(1)


    model = UNetModel(
        image_size=args.img_width,
        in_channels=model_input_channels,
        model_channels=args.num_channels,
        out_channels=C,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_resolutions_tuple,
        dropout=args.dropout,
        channel_mult=channel_mult,
        use_checkpoint=False,
        use_fp16=args.fp16,
        num_heads=args.num_heads,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(DEVICE)

    # Initialize EMA model if enabled
    ema_model = None
    if args.ema_model_saving:
        ema_model = copy.deepcopy(model).to(DEVICE)
        ema_model.eval()
        print("Initialized EMA model")

    # - Flow Matcher -
    cfm_class_map = {
        "vanilla": ConditionalFlowMatcher,
    }
    if args.flow_matching_method not in cfm_class_map:
        raise ValueError(f"Invalid flow matching method: {args.flow_matching_method}")
    flow_matcher = cfm_class_map[args.flow_matching_method](sigma=args.cfm_sigma)
    print(
        f"Using Flow Matcher: {args.flow_matching_method} with sigma={args.cfm_sigma}"
    )

    # - Optimizer and Scheduler -
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Initialize scheduler based on type
    scheduler = None
    if args.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.lr_plateau_min_lr,
            threshold=args.lr_plateau_threshold,
            verbose=True,
        )
        print(
            f"Initialized ReduceLROnPlateau scheduler with factor={args.lr_plateau_factor}, patience={args.lr_plateau_patience}"
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_plateau_min_lr
        )
        print("Initialized CosineAnnealingLR scheduler")
    elif args.scheduler_type != "none":
        raise ValueError(f"Invalid scheduler type: {args.scheduler_type}")

    # - Loss -
    criterion = nn.MSELoss(reduction="none")

    # - Mixed Precision Scaler -
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    # - Early Stopping -
    early_stopping = None
    if args.early_stopping and len(val_split) > 0:  # Only enable if val set exists
        # Determine metric direction based on the chosen metric
        metric_direction = (
            "minimize" if args.early_stopping_metric in ["val_loss"] else "maximize"
        )

        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            verbose=args.early_stopping_verbose,
            delta=args.early_stopping_delta,
            path=os.path.join(checkpoints_folder, "early_stopping_best_model.pt"),
            metric_direction=metric_direction,  # Set direction based on metric
            save_scheduler=scheduler,  # Pass scheduler to save its state
        )
        print(
            f"Initialized early stopping with patience={args.early_stopping_patience}, "
            f"metric={args.early_stopping_metric}, direction={metric_direction}"
        )
    elif args.early_stopping:
        print("Warning: Early stopping enabled but validation set is empty. Disabling.")
        args.early_stopping = False

    # - Metrics Initialization -
    psnr_metric = None
    ssim_metric = None
    if args.eval_partial_metrics_per_epoch and PeakSignalNoiseRatio is not None:
        # PSNR/SSIM expect inputs in [0, 1] range, set data_range=1.0
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        print("Initialized PSNR and SSIM metrics.")

    # - WandB -
    if args.enable_wandb:
        # Ensure API key is set via environment variable WANDB_API_KEY
        try:
            wandb.init(project="deblur_cfm", config=vars(args), name=RUN_ID)
            wandb.watch(model, log="gradients", log_freq=100, log_graph=True)
        except Exception as e:
            print(f"Could not initialize WandB: {e}. Disabling WandB.")
            args.enable_wandb = False

    # - Training Loop -
    global_step = 0
    best_val_loss = float("inf")
    print("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False
        )
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(progress_bar):
            blurred_images, sharp_images = batch
            x_cond = blurred_images.to(DEVICE)
            x1 = sharp_images.to(DEVICE)
            current_batch_size = x1.shape[0]

            # Only zero gradients at the start of accumulation
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)  # More efficient zeroing

            # Sample noise (prior) - typically standard Gaussian for CFM
            x0_noise = torch.randn_like(x1)

            with torch.amp.autocast(
                device_type=DEVICE,
                dtype=torch.float16 if args.fp16 else torch.float32,
                enabled=args.fp16,
            ):
                # - CFM Step -
                # Sample time t, intermediate state x_t, and target vector field u_t
                t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                    x0=x0_noise, x1=x1  # Flow from noise x0 to sharp image x1
                )

                # - Model Prediction -
                # The model predicts the vector field v_t(x_t, t, x_cond) that should match u_t
                model_input = torch.cat([x_t, x_cond], dim=1)


                # Pass continuous time t directly
                v_t_pred = model(x=model_input, t=t)

                # - Loss Calculation -
                # Calculate MSE loss between predicted vector field v_t_pred and target u_t
                # Scale loss by gradient accumulation steps
                loss = (
                    criterion(v_t_pred, u_t).mean() / args.gradient_accumulation_steps
                )  # Average over all dimensions and batch

            # - End Autocast -

            scaler.scale(loss).backward()

            # Only perform optimizer step after accumulating gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                if args.grad_clip_val is not None and args.grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_val
                    )

                scaler.step(optimizer)
                scaler.update()

                # - Update EMA model -
                if args.ema_model_saving:
                    ema(model, ema_model, args.ema_model_saving_decay)

            # - Logging -
            # Scale loss back for logging
            loss_item = loss.item() * args.gradient_accumulation_steps
            total_train_loss += loss_item * current_batch_size
            progress_bar.set_postfix(loss=f"{loss_item:.4f}")

            if args.enable_wandb and (global_step % 100 == 0):
                wandb.log({"train_step_loss": loss_item}, step=global_step)

            global_step += 1

        # - End of Epoch -
        avg_train_loss = (
            total_train_loss / len(train_split) if len(train_split) > 0 else 0.0
        )

        # - Validation Phase -
        if len(val_loader) > 0:
            model.eval()
            if ema_model is not None:
                ema_model.eval() # Ensure EMA model is in eval mode

            total_val_loss = 0.0
            val_psnr_accum = 0.0
            val_ssim_accum = 0.0
            val_samples_count_for_metrics = 0 # Track samples used for PSNR/SSIM

            with torch.no_grad():
                # - Calculate Validation Loss -
                for batch in tqdm(val_loader, desc="Validation Loss", leave=False):
                    blurred_images_val, sharp_images_val = batch
                    x_cond_val = blurred_images_val.to(DEVICE)
                    x1_val = sharp_images_val.to(DEVICE)
                    current_batch_size_val = x1_val.shape[0]
                    x0_noise_val = torch.randn_like(x1_val)

                    with torch.amp.autocast(
                        device_type=DEVICE,
                        dtype=torch.float16 if args.fp16 else torch.float32,
                        enabled=args.fp16,
                    ):
                        t_val, x_t_val, u_t_val = (
                            flow_matcher.sample_location_and_conditional_flow(
                                x0=x0_noise_val, x1=x1_val
                            )
                        )
                        model_input_val = torch.cat([x_t_val, x_cond_val], dim=1)
                        # Use the main model (or EMA model if ema model saving is enabled)
                        model_to_use_for_loss = (
                            ema_model if args.ema_model_saving else model
                        )
                        v_t_val_pred = model_to_use_for_loss(x=model_input_val, t=t_val)
                        val_loss_item = criterion(v_t_val_pred, u_t_val).mean()

                    total_val_loss += val_loss_item.item() * current_batch_size_val

                avg_val_loss = total_val_loss / len(val_split)

                # - Generate Validation Samples & Calculate Metrics -
                if (
                    args.eval_partial_metrics_per_epoch
                    and fixed_val_blurred is not None
                    and actual_val_samples > 0
                ):
                    print(
                        f"Generating {actual_val_samples} validation samples via ODE (batches of {val_batch_size})..."
                    )
                    all_generated_samples = []
                    num_sampling_batches = (
                        actual_val_samples + val_batch_size - 1
                    ) // val_batch_size
                    ode_time_span = torch.tensor(
                        [0.0, 1.0], device=DEVICE
                    )  # Integrate from t=0 (noise) to t=1 (image)

                    ode_options = None
                    if args.ode_solver in ["euler", "rk4", "midpoint"]:
                        ode_options = {"step_size": 1.0 / args.ode_steps}

                    # Use EMA model for generation if enabled, otherwise use main model
                    inference_model = ema_model if args.ema_model_saving else model
                    inference_model.eval()  # Ensure model is in eval mode

                    for i in tqdm(
                        range(num_sampling_batches), desc="ODE Sampling", leave=False
                    ):
                        start_idx = i * val_batch_size
                        end_idx = min((i + 1) * val_batch_size, actual_val_samples)
                        current_sampling_batch_size = end_idx - start_idx
                        if current_sampling_batch_size <= 0:
                            continue

                        condition_batch = fixed_val_blurred[start_idx:end_idx]
                        # Initial state is noise x0 ~ N(0, I)
                        x0_noise_batch = torch.randn(
                            (
                                current_sampling_batch_size,
                                C,
                                args.img_height,
                                args.img_width,
                            ),
                            device=DEVICE,
                        )

                        # - ODE Integration -
                        # Wrap the inference model for the ODE solver
                        flow_dynamics_func = FlowDynamics(
                            model=inference_model,
                            condition=condition_batch,
                        )

                        # Detach parameters if not using adjoint method for efficiency
                        solution = odeint(
                            flow_dynamics_func,
                            x0_noise_batch,
                            ode_time_span,
                            method=args.ode_solver,
                            options=ode_options,
                            atol=args.ode_atol,
                            rtol=args.ode_rtol,
                        )
                        generated_batch = solution[
                            -1
                        ]  # State at t=1 is the generated image
                        all_generated_samples.append(generated_batch)

                    generated_samples_neg1_1 = torch.cat(all_generated_samples, dim=0)

                    # - Calculate PSNR/SSIM -
                    if psnr_metric is not None and ssim_metric is not None:
                        # Convert generated and GT images from [-1, 1] to [0, 1] for metrics
                        generated_01 = (generated_samples_neg1_1.clamp(-1, 1) + 1) / 2.0
                        sharp_gt_01 = (fixed_val_sharp.clamp(-1, 1) + 1) / 2.0

                        # Update metrics batch by batch or all at once
                        # Using all at once for simplicity here
                        current_psnr = psnr_metric(generated_01, sharp_gt_01)
                        current_ssim = ssim_metric(generated_01, sharp_gt_01)
                        val_psnr_accum = current_psnr.item()
                        val_ssim_accum = current_ssim.item()
                        print(
                            f"[Metrics] Epoch {epoch+1}: PSNR = {val_psnr_accum:.4f}, SSIM = {val_ssim_accum:.4f}"
                        )

                    # - Save Samples -
                    save_val_samples(
                        epoch,
                        fixed_val_blurred,  # Condition [-1, 1]
                        fixed_val_sharp,  # Ground Truth [-1, 1]
                        generated_samples_neg1_1,  # Generated [-1, 1]
                        samples_folder,
                    )

            # - Logging after validation -
            print(
                f"Epoch {epoch+1}/{args.epochs} Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            log_dict = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],  # Get current LR
            }
            if args.eval_partial_metrics_per_epoch and psnr_metric is not None:
                log_dict["val_psnr"] = val_psnr_accum
                log_dict["val_ssim"] = val_ssim_accum

            if args.enable_wandb:
                wandb.log(log_dict, step=global_step)  # Log epoch summary

            # - Scheduler Step -
            if scheduler is not None:
                if args.scheduler_type == "plateau":
                    scheduler.step(avg_val_loss)
                elif args.scheduler_type == "cosine":
                    scheduler.step()

            # - Checkpoint Saving (Best Validation Loss) -
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_path = os.path.join(checkpoints_folder, "deblur_cfm_unet_best.pt")
                model_state_to_save = (
                    ema_model.state_dict()
                    if args.ema_model_saving
                    else model.state_dict()
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_state_dict": model_state_to_save,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,  # Save the loss that corresponds to this checkpoint
                        "val_psnr": (
                            val_psnr_accum if "val_psnr_accum" in locals() else None
                        ),
                        "val_ssim": (
                            val_ssim_accum if "val_ssim_accum" in locals() else None
                        ),
                        "args": vars(args),
                        "img_dims": IMG_DIMS,
                    },
                    ckpt_path,
                )
                print(f"-> Validation loss improved; checkpoint saved to {ckpt_path}")

            # - Early Stopping Check -
            if args.early_stopping:
                # Select the appropriate metric based on the argument
                metric_for_early_stop = None
                if args.early_stopping_metric == "val_loss":
                    metric_for_early_stop = avg_val_loss
                elif (
                    args.early_stopping_metric == "psnr"
                    and "val_psnr_accum" in locals()
                ):
                    metric_for_early_stop = val_psnr_accum
                elif (
                    args.early_stopping_metric == "ssim"
                    and "val_ssim_accum" in locals()
                ):
                    metric_for_early_stop = val_ssim_accum

                if metric_for_early_stop is not None:
                    # Pass the EMA model to save if enabled, otherwise the base model
                    model_for_early_stop = ema_model if args.ema_model_saving else model
                    early_stopping(
                        metric_for_early_stop,
                        model_for_early_stop,
                        optimizer,
                        epoch,
                        global_step,
                    )
                    if early_stopping.early_stop:
                        print(
                            f"Early stopping triggered based on {args.early_stopping_metric}"
                        )
                        break
                else:
                    print(
                        f"Warning: Could not compute {args.early_stopping_metric} for early stopping"
                    )
        else:  # No validation loader
            print(
                f"Epoch {epoch+1}/{args.epochs} Summary - Train Loss: {avg_train_loss:.4f} (No validation)"
            )
            if args.enable_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "avg_train_loss": avg_train_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            # Save latest model if no validation
            ckpt_path = os.path.join(checkpoints_folder, "deblur_cfm_unet_latest.pt")
            model_state_to_save = (
                ema_model.state_dict() if args.ema_model_saving else model.state_dict()
            )
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model_state_to_save}, ckpt_path
            )

    # - End Training Loop -
    print("Training Finished.")
    if len(val_split) > 0:
        print(f"Best validation loss achieved: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
