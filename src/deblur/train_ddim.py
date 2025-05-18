import sys
import os
import shutil

sys.path.append(os.getcwd())  # Add the current working directory to the path

import torch
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
from src.models.unet import UNetModel
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from src.deblur.ddim_sampler import DDIMSampler, get_beta_schedule, get_alpha_schedule
from src.deblur.datasets.deblur import make_deblur_splits
from src.utils.early_stopping import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDIM for Image Deblurring")
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
        help="Directory containing sharp (ground truth) images",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=272,
        help="Image height (MUST be compatible with UNet downsampling, e.g., divisible by 8 or 16)",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=480,
        help="Image width (MUST be compatible with UNet downsampling, e.g., divisible by 8 or 16)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing an optimizer step",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (T)",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        help="Beta schedule (only linear supported)",
    )
    parser.add_argument(
        "--beta_start", type=float, default=1e-4, help="Beta schedule start value"
    )
    parser.add_argument(
        "--beta_end", type=float, default=2e-2, help="Beta schedule end value"
    )
    # - Model Specific Args -
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
        "--channel_mult",
        type=str,
        default="1,2,2,2",
        help="Channel multipliers for UNet levels (comma-separated)",
    )
    parser.add_argument(
        "--attention_resolutions",
        type=str,
        default="8,16",
        help="Resolutions for attention layers (comma-separated)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    # - Sampling parameters for validation -
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=5,
        help="Number of images to generate during validation",
    )
    # - Validation Batch Size -
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation sampling (defaults to training batch_size)",
    )
    parser.add_argument(
        "--val_num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM steps for validation sampling",
    )
    parser.add_argument(
        "--val_eta", type=float, default=0.0, help="DDIM eta for validation sampling"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu'). Auto-detects if None.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers for DataLoader (default: 12)",
    )
    parser.add_argument(
        "--run_string", type=str, default="deblur_base", help="Run String"
    )
    parser.add_argument(
        "--enable_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging",
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
        default=True,
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

    # - Metrics Evaluation -
    parser.add_argument(
        "--eval_partial_metrics_per_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate partial metrics  at the end of each epoch (slow)",
    )

    return parser.parse_args()


# - Helper to save validation images -
def save_val_samples(epoch, blurred_cond, sharp_gt, generated, sample_dir):
    """Saves comparison images: blurred input, generated output, ground truth sharp."""
    epoch_dir = os.path.join(sample_dir, f"epoch_{epoch+1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Ensure tensors are on CPU and detach from graph
    blurred_cond = blurred_cond.cpu().detach()
    sharp_gt = sharp_gt.cpu().detach()
    generated = generated.cpu().detach()

    # Scale images from [-1, 1] to [0, 1] for saving
    blurred_cond = (blurred_cond + 1) / 2.0
    sharp_gt = (sharp_gt + 1) / 2.0
    generated = (generated + 1) / 2.0

    num_samples = generated.shape[0]
    for i in range(num_samples):
        # Combine input, output, ground_truth side-by-side
        comparison_grid = torch.cat(
            [blurred_cond[i], generated[i], sharp_gt[i]], dim=2
        )  # Cat width-wise
        vutils.save_image(
            comparison_grid, os.path.join(epoch_dir, f"val_sample_{i:02d}.png")
        )

    print(f"Saved {num_samples} validation samples to {epoch_dir}")


def ema(model, ema_model, decay):
    """Update the EMA model weights using exponential moving average."""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    args = parse_args()
    DEVICE = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {DEVICE}")

    # Determine validation batch size
    val_batch_size = (
        args.val_batch_size if args.val_batch_size is not None else args.batch_size
    )
    print(f"Using validation sampling batch size: {val_batch_size}")

    # - Setup Artifacts/Checkpoints/Samples Directories -
    FOLDER = "src/deblur"
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
    samples_folder = os.path.join(
        run_artifacts_folder, "samples"
    ) # Base folder for samples

    os.makedirs(run_artifacts_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(samples_folder, exist_ok=True) # Main samples folder for the run
    print(f"Run artifacts will be saved in: {run_artifacts_folder}")

    # - Dataset -
    IMG_DIMS = (args.img_height, args.img_width)
    C = 3
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_DIMS),
            transforms.ToTensor(), # Converts to [0, 1] range, CxHxW
            transforms.Lambda(lambda t: (t * 2) - 1), # Scale to [-1, 1] range
        ]
    )

    print(
        f"Loading dataset from: Blurred='{args.blurred_dir}', Sharp='{args.sharp_dir}'"
    )
    try:
        train_split, val_split = make_deblur_splits(
            args.blurred_dir, args.sharp_dir, transform, val_ratio=0.2
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"Error creating dataset splits: {e}")
        sys.exit(1)

    train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get a fixed batch from validation set for consistent sampling visualization
    fixed_val_batch = None
    fixed_val_blurred = None
    fixed_val_sharp = None
    actual_val_samples = 0  # Track how many samples we actually have/use
    if len(val_split) > 0:
        # Collect enough samples from validation set for visualization
        num_to_collect = args.val_num_samples
        collected_blurred = []
        collected_sharp = []
        # Iterate through val_loader to get enough unique samples
        # This ensures we use different images if val_num_samples > val_loader batch size
        temp_val_loader = DataLoader(
            val_split, batch_size=args.batch_size, shuffle=False
        )  # Use training batch size here
        for batch in temp_val_loader:
            blurred, sharp = batch
            needed = num_to_collect - len(collected_blurred)
            if needed <= 0:
                break
            count = min(needed, blurred.shape[0])
            collected_blurred.append(blurred[:count])
            collected_sharp.append(sharp[:count])
            if len(collected_blurred) >= num_to_collect:
                break # Exit if we collected enough

        if collected_blurred:
            fixed_val_blurred = torch.cat(collected_blurred, dim=0)[:num_to_collect].to(
                DEVICE
            )
            fixed_val_sharp = torch.cat(collected_sharp, dim=0)[:num_to_collect].to(
                DEVICE
            )
            actual_val_samples = fixed_val_blurred.shape[0] # Actual number collected
            if actual_val_samples < args.val_num_samples:
                print(
                    f"Warning: Could only collect {actual_val_samples} samples from validation set for visualization (requested {args.val_num_samples})."
                )
            print(
                f"Using {actual_val_samples} samples from validation set for visualization."
            )
        else:
            print("Warning: Could not retrieve any samples from validation loader.")

    else:
        print("Warning: Validation set is empty. Cannot generate validation samples.")

    # - Model -
    model_input_channels = 6

    channel_mult = tuple(map(int, args.channel_mult.split(",")))
    try:
        # The --attention_resolutions argument should now be a comma-separated
        # string of actual downsample factors (e.g., "4,8,16").
        if args.attention_resolutions.strip():
            attention_resolutions_list = [
                int(ds_factor.strip()) for ds_factor in args.attention_resolutions.split(',') if ds_factor.strip()
            ]
            for ds_factor in attention_resolutions_list:
                if ds_factor <= 0 or (ds_factor > 1 and (ds_factor & (ds_factor - 1)) != 0 and ds_factor != 1): 
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
        use_fp16=False,
        num_heads=args.num_heads,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(DEVICE)

    # Initialize EMA model if enabled
    ema_model = None
    if args.ema_model_saving:
        ema_model = copy.deepcopy(model)
        print("Initialized EMA model")

    # - Metrics Initialization -
    psnr_metric = None
    ssim_metric = None
    if args.eval_partial_metrics_per_epoch and PeakSignalNoiseRatio is not None:
        # PSNR/SSIM expect inputs in [0, 1] range, set data_range=1.0
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        print("Initialized PSNR and SSIM metrics.")

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

    # - Early Stopping -
    early_stopping = None
    if args.early_stopping and len(val_split) > 0: # Only enable if val set exists
        # Determine metric direction based on the chosen metric
        metric_direction = (
            "minimize" if args.early_stopping_metric in ["val_loss"] else "maximize"
        )

        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            verbose=args.early_stopping_verbose,
            delta=args.early_stopping_delta,
            path=os.path.join(checkpoints_folder, "early_stopping_best_model.pt"),
            metric_direction=metric_direction, # Set direction based on metric
            save_scheduler=scheduler, # Pass scheduler to save its state
        )
        print(
            f"Initialized early stopping with patience={args.early_stopping_patience}, "
            f"metric={args.early_stopping_metric}, direction={metric_direction}"
        )
    elif args.early_stopping:
        print("Warning: Early stopping enabled but validation set is empty. Disabling.")
        args.early_stopping = False

    # - Noise Schedule -
    betas = get_beta_schedule(
        args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=args.num_timesteps,
    )
    alphas, alphas_cumprod = get_alpha_schedule(betas)
    alphas_cumprod = torch.from_numpy(alphas_cumprod).float().to(DEVICE)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # - WandB Initialization -
    if args.enable_wandb:
        wandb.init(
            project="dl_deblur",
            config={
                k: str(v) if isinstance(v, (tuple, list)) else v
                for k, v in vars(args).items()
            }, # Log args
            name=RUN_ID, # Use generated run ID for wandb run name
        )
        wandb.watch(model, log="gradients", log_freq=100) # Watch model gradients

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
            blurred_images = blurred_images.to(DEVICE)
            sharp_images = sharp_images.to(DEVICE)
            batch_size = sharp_images.shape[0]

            # Only zero gradients at the start of accumulation
            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            t = torch.randint(
                0, args.num_timesteps, (batch_size,), device=DEVICE
            ).long()
            noise = torch.randn_like(sharp_images)

            sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alpha_t * sharp_images + sqrt_one_minus_alpha_t * noise

            model_input = x_t
            if blurred_images.shape[2:] != x_t.shape[2:]:
                raise ValueError("Blurred and sharp image dimensions mismatch.")
            model_input = torch.cat([x_t, blurred_images], dim=1)

            predicted_noise = model(x=model_input, t=t)
            # Scale loss by gradient accumulation steps
            loss = F.mse_loss(noise, predicted_noise) / args.gradient_accumulation_steps
            loss.backward()

            # Only perform optimizer step after accumulating gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Update EMA model if enabled
                if args.ema_model_saving:
                    ema(model, ema_model, args.ema_model_saving_decay)

            # Log training loss to wandb (scale loss back for logging)
            if args.enable_wandb:
                wandb.log(
                    {
                        "train_loss": loss.item() * args.gradient_accumulation_steps,
                    },
                    step=global_step,
                )

            # Scale loss back for logging and accumulation
            total_train_loss += (
                loss.item() * args.gradient_accumulation_steps * batch_size
            )  # Weighted average by batch size
            progress_bar.set_postfix(
                loss=loss.item() * args.gradient_accumulation_steps
            )
            global_step += 1

        avg_train_loss = total_train_loss / len(train_split) # Average over samples

        # - Validation -
        model.eval()
        if ema_model is not None:
            ema_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc="Validation", leave=False)
            ):
                blurred_images_val, sharp_images_val = batch
                blurred_images_val = blurred_images_val.to(DEVICE)
                sharp_images_val = sharp_images_val.to(DEVICE)
                batch_size_val = sharp_images_val.shape[0]

                t_val = torch.randint(
                    0, args.num_timesteps, (batch_size_val,), device=DEVICE
                ).long()
                noise_val = torch.randn_like(sharp_images_val)

                sqrt_alpha_t_val = sqrt_alphas_cumprod[t_val].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_t_val = sqrt_one_minus_alphas_cumprod[t_val].view(
                    -1, 1, 1, 1
                )
                x_t_val = (
                    sqrt_alpha_t_val * sharp_images_val
                    + sqrt_one_minus_alpha_t_val * noise_val
                )

                model_input_val = x_t_val
                if blurred_images_val.shape[2:] != x_t_val.shape[2:]:
                    raise ValueError(
                        "Validation: Blurred/sharp dimensions mismatch."
                    )
                model_input_val = torch.cat([x_t_val, blurred_images_val], dim=1)

                predicted_noise_val = model(
                    x=model_input_val, t=t_val
                ) 
                val_loss_item = F.mse_loss(noise_val, predicted_noise_val)
                total_val_loss += val_loss_item.item() * batch_size_val

            if len(val_split) > 0:
                avg_val_loss = total_val_loss / len(val_split)
            else:
                avg_val_loss = 0 # Handle empty validation set

            print(
                f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # - Generate Validation Samples & Calculate Metrics -
            if (
                args.eval_partial_metrics_per_epoch
                and fixed_val_blurred is not None
                and actual_val_samples > 0
            ):
                print(
                    f"Generating {actual_val_samples} validation samples via DDIM (batches of {val_batch_size})..."
                )
                all_generated_samples = []
                num_sampling_batches = (
                    actual_val_samples + val_batch_size - 1
                ) // val_batch_size

                # Use EMA model for generation if enabled, otherwise use main model
                inference_model = ema_model if args.ema_model_saving else model
                inference_model.eval()

                # Initialize DDIM sampler for validation
                val_sampler = DDIMSampler(
                    inference_model,
                    beta_schedule=args.beta_schedule,
                    beta_start=args.beta_start,
                    beta_end=args.beta_end,
                    num_diffusion_timesteps=args.num_timesteps,
                    device=DEVICE,
                )

                for i in tqdm(
                    range(num_sampling_batches), desc="DDIM Sampling", leave=False
                ):
                    start_idx = i * val_batch_size
                    end_idx = min((i + 1) * val_batch_size, actual_val_samples)
                    current_sampling_batch_size = end_idx - start_idx
                    if current_sampling_batch_size <= 0:
                        continue

                    condition_batch = fixed_val_blurred[start_idx:end_idx]

                    # Generate samples using DDIM
                    generated_batch = val_sampler.sample(
                        batch_size=current_sampling_batch_size,
                        image_size=(C, args.img_height, args.img_width),
                        num_inference_steps=args.val_num_inference_steps,
                        eta=args.val_eta,
                        condition=condition_batch,
                    )
                    all_generated_samples.append(generated_batch)

                generated_samples_neg1_1 = torch.cat(all_generated_samples, dim=0)

                # - Calculate PSNR/SSIM -
                if psnr_metric is not None and ssim_metric is not None:
                    # Convert generated and GT images from [-1, 1] to [0, 1] for metrics
                    generated_01 = (generated_samples_neg1_1.clamp(-1, 1) + 1) / 2.0
                    sharp_gt_01 = (fixed_val_sharp.clamp(-1, 1) + 1) / 2.0

                    # Update metrics batch by batch or all at once
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
                    fixed_val_blurred, # Condition [-1, 1]
                    fixed_val_sharp, # Ground Truth [-1, 1]
                    generated_samples_neg1_1, # Generated [-1, 1]
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
                "learning_rate": optimizer.param_groups[0]["lr"], # Get current LR
            }
            if args.eval_partial_metrics_per_epoch and psnr_metric is not None:
                log_dict["val_psnr"] = val_psnr_accum
                log_dict["val_ssim"] = val_ssim_accum

            if args.enable_wandb:
                wandb.log(log_dict, step=global_step) # Log epoch summary

            # - Scheduler Step -
            if scheduler is not None:
                if args.scheduler_type == "plateau":
                    scheduler.step(avg_val_loss)
                elif args.scheduler_type == "cosine":
                    scheduler.step()

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

        # - Save Checkpoint -
        if avg_val_loss < best_val_loss and len(val_split) > 0:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(checkpoints_folder, "deblur_unet_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        ema_model.state_dict()
                        if args.ema_model_saving
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
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
            print(f"-> Validation improved; checkpoint saved to {ckpt_path}")

    print("Training Finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
