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

# - DDP Imports -
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp(rank, world_size, local_rank):
    """Initializes the distributed environment."""
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method="env://"
    )
    torch.cuda.set_device(local_rank)
    dist.barrier()  # Ensure all processes are synchronized after setup
    print(
        f"DDP Initialized: Rank {rank}/{world_size} on GPU {torch.cuda.current_device()} (Local Rank: {local_rank})"
    )


def cleanup_ddp():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Cleaned up DDP.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DDIM for Image Deblurring (DDP)"
    )
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
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-GPU training batch size"
    )
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
        default="16",
        help="Resolutions for attention layers (comma-separated)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=5,
        help="Number of images to generate during validation (on rank 0)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation sampling (defaults to training batch_size, used on rank 0)",
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
    # DDP local_rank will be provided by torchrun
    # parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detects if None.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers for DataLoader per GPU (default: 12)",
    )
    parser.add_argument(
        "--run_string", type=str, default="deblur_ddp", help="Run String"
    )
    parser.add_argument(
        "--enable_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging (only on rank 0)",
    )
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
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping (monitored on rank 0)",
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
        help="Verbose early stopping (on rank 0)",
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
    parser.add_argument(
        "--eval_partial_metrics_per_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate partial metrics at the end of each epoch (on rank 0, slow)",
    )
    parser.add_argument(
        "--preload_model",
        type=str,
        default=None,
        help="Path to a .pt model checkpoint to preload (e.g., one saved by early stopping or other training runs).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for DDP, set by launch script",
    )
    return parser.parse_args()


def save_val_samples(epoch, blurred_cond, sharp_gt, generated, sample_dir, rank):
    """Saves comparison images: blurred input, generated output, ground truth sharp."""
    if rank != 0:
        return

    epoch_dir = os.path.join(sample_dir, f"epoch_{epoch+1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    blurred_cond = blurred_cond.cpu().detach()
    sharp_gt = sharp_gt.cpu().detach()
    generated = generated.cpu().detach()

    blurred_cond = (blurred_cond + 1) / 2.0
    sharp_gt = (sharp_gt + 1) / 2.0
    generated = (generated + 1) / 2.0

    num_samples = generated.shape[0]
    for i in range(num_samples):
        comparison_grid = torch.cat([blurred_cond[i], generated[i], sharp_gt[i]], dim=2)
        vutils.save_image(
            comparison_grid, os.path.join(epoch_dir, f"val_sample_{i:02d}.png")
        )
    print(f"Rank {rank}: Saved {num_samples} validation samples to {epoch_dir}")


def ema(model, ema_model, decay):
    """Update the EMA model weights using exponential moving average."""
    # Access underlying model if DDP wrapped
    model_params = (
        model.module.parameters() if isinstance(model, DDP) else model.parameters()
    )
    ema_model_params = (
        ema_model.module.parameters()
        if isinstance(ema_model, DDP)
        else ema_model.parameters()
    )

    with torch.no_grad():
        for param, ema_param in zip(model_params, ema_model_params):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    args = parse_args()

    # - DDP Setup -

    # Infer world size and rank from environment variables if available
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))  # Global rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Local rank, GPU ID

    if world_size > 1:
        print(
            f"Initializing DDP for Rank {rank}, Local Rank {local_rank}, World Size {world_size}"
        )
        setup_ddp(rank, world_size, local_rank)  # Pass local_rank
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Handle single GPU or CPU case
        if torch.cuda.is_available():
            device = torch.device("cuda")
            local_rank = 0
        else:
            device = torch.device("cpu")
            local_rank = 0
        print(f"Running on single device: {device}")

    DEVICE = device  # Main device for this process

    if rank == 0:
        print(f"Using device: {DEVICE} (Local Rank: {local_rank})")

    val_batch_size = (
        args.val_batch_size if args.val_batch_size is not None else args.batch_size
    )
    if rank == 0:
        print(f"Using validation sampling batch size (on rank 0): {val_batch_size}")

    # - Setup Artifacts/Checkpoints/Samples Directories (Rank 0 only) -
    run_artifacts_folder = None
    checkpoints_folder = None
    samples_folder = None
    RUN_ID = None

    if rank == 0:
        FOLDER = "src/deblur"
        # Generate RUN_ID on rank 0
        base_run_id = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + args.run_string
        )
        name_part = namegenerator.gen()
        RUN_ID = base_run_id + "_" + name_part

        artifacts_base_folder = os.path.join(FOLDER, "artifacts", args.run_string)
        run_artifacts_folder = os.path.join(artifacts_base_folder, RUN_ID)
        checkpoints_folder = os.path.join(run_artifacts_folder, "checkpoints")
        samples_folder = os.path.join(run_artifacts_folder, "samples")

        os.makedirs(run_artifacts_folder, exist_ok=True)
        os.makedirs(checkpoints_folder, exist_ok=True)
        os.makedirs(samples_folder, exist_ok=True)
        print(f"Run artifacts will be saved in: {run_artifacts_folder}")

    if world_size > 1:
        # Broadcast RUN_ID from rank 0 to all other processes
        run_id_container = [RUN_ID] if rank == 0 else [None]
        dist.broadcast_object_list(run_id_container, src=0)
        if rank != 0:
            RUN_ID = run_id_container[0]
        dist.barrier()  # Ensure all have RUN_ID before proceeding

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

    if rank == 0:
        print(
            f"Loading dataset from: Blurred='{args.blurred_dir}', Sharp='{args.sharp_dir}'"
        )

    try:
        # Create full datasets first, then subset with samplers
        full_train_split, full_val_split = make_deblur_splits(
            args.blurred_dir, args.sharp_dir, transform, val_ratio=0.2
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        if rank == 0:
            print(f"Error creating dataset splits: {e}")
        sys.exit(1)

    train_sampler = (
        DistributedSampler(
            full_train_split, num_replicas=world_size, rank=rank, shuffle=True
        )
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            full_val_split, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        full_train_split,
        batch_size=args.batch_size,  # per-GPU batch size
        shuffle=(train_sampler is None),  # Shuffle if not using sampler
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # For validation loss calculation, use the distributed sampler
    val_loader = DataLoader(
        full_val_split,
        batch_size=args.batch_size,  # Per-GPU for distributed validation loss
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    if rank == 0:
        print(
            f"Train batches per GPU: {len(train_loader)}, Val batches per GPU: {len(val_loader)}"
        )
        print(
            f"Total train samples: {len(full_train_split)}, Total val samples: {len(full_val_split)}"
        )

    # Get a fixed batch from validation set for consistent sampling visualization (Rank 0 only)
    fixed_val_blurred = None
    fixed_val_sharp = None
    actual_val_samples = 0
    if rank == 0 and len(full_val_split) > 0:
        num_to_collect = args.val_num_samples
        collected_blurred = []
        collected_sharp = []
        # Use a temporary loader on rank 0 to get these specific samples
        temp_val_data_for_fixed_samples = DataLoader(
            full_val_split, batch_size=args.batch_size, shuffle=False
        )
        for batch_idx, batch in enumerate(temp_val_data_for_fixed_samples):
            if (
                len(collected_blurred) * args.batch_size >= num_to_collect
                and len(collected_blurred) > 0
            ):
                break  # Optimization: if we collected enough in previous batches
            blurred, sharp = batch
            needed = num_to_collect - sum(b.shape[0] for b in collected_blurred)
            if needed <= 0:
                break
            count = min(needed, blurred.shape[0])
            collected_blurred.append(blurred[:count])
            collected_sharp.append(sharp[:count])
            if sum(b.shape[0] for b in collected_blurred) >= num_to_collect:
                break

        if collected_blurred:
            fixed_val_blurred = torch.cat(collected_blurred, dim=0)[:num_to_collect].to(
                DEVICE
            )
            fixed_val_sharp = torch.cat(collected_sharp, dim=0)[:num_to_collect].to(
                DEVICE
            )
            actual_val_samples = fixed_val_blurred.shape[0]
            if actual_val_samples < args.val_num_samples:
                print(
                    f"Warning (Rank 0): Could only collect {actual_val_samples} samples from validation set for visualization (requested {args.val_num_samples})."
                )
            print(
                f"Rank 0: Using {actual_val_samples} samples from validation set for visualization."
            )
        else:
            print(
                "Warning (Rank 0): Could not retrieve any samples from validation loader for fixed visualization."
            )
    elif rank == 0:
        print(
            "Warning (Rank 0): Validation set is empty. Cannot generate validation samples."
        )

    # - Model -
    model_input_channels = 6
    channel_mult = tuple(map(int, args.channel_mult.split(",")))
    attention_resolutions_tuple = tuple()
    if rank == 0:  # Parse complex args on rank 0 and broadcast if needed, or all parse
        try:
            if args.attention_resolutions.strip():
                attention_resolutions_list = [
                    int(ds_factor.strip())
                    for ds_factor in args.attention_resolutions.split(",")
                    if ds_factor.strip()
                ]
                for ds_factor in attention_resolutions_list:
                    if ds_factor <= 0 or (
                        ds_factor > 1
                        and (ds_factor & (ds_factor - 1)) != 0
                        and ds_factor != 1
                    ):
                        print(
                            f"Warning: Attention resolution ds_factor {ds_factor} is not a power of 2."
                        )
            else:
                attention_resolutions_list = []
            attention_resolutions_tuple = tuple(attention_resolutions_list)
            print(
                f"Using attention at downsample factors (ds): {attention_resolutions_tuple}"
            )
        except ValueError as e:
            print(
                f"Error: Invalid attention resolutions format '{args.attention_resolutions}'. Error: {e}"
            )
            if world_size > 1:
                dist.barrier()  # ensure other processes don't hang
            sys.exit(1)

    # Create model on CPU first to ensure clean state loading from checkpoint (which is loaded to CPU)
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
    )

    loaded_optimizer_state_dict = None
    loaded_scheduler_state_dict = None
    loaded_global_step = None

    if args.preload_model:
        if os.path.exists(args.preload_model):
            if rank == 0:
                print(f"Rank {rank}: Loading checkpoint from {args.preload_model}...")
            try:
                checkpoint = torch.load(args.preload_model, map_location='cpu') # Load to CPU

                model.load_state_dict(checkpoint['model_state_dict'])
                if rank == 0:
                    print(f"Rank {rank}: Model state loaded to CPU model from checkpoint.")

                if 'optimizer_state_dict' in checkpoint:
                    loaded_optimizer_state_dict = checkpoint['optimizer_state_dict']
                    if rank == 0:
                        print(f"Rank {rank}: Optimizer state found in checkpoint, will load after optimizer init.")
                if 'scheduler_state_dict' in checkpoint:
                    loaded_scheduler_state_dict = checkpoint['scheduler_state_dict']
                    if rank == 0:
                        print(f"Rank {rank}: Scheduler state found in checkpoint, will load after scheduler init.")
                if 'global_step' in checkpoint:
                    loaded_global_step = checkpoint['global_step']
                    if rank == 0:
                        print(f"Rank {rank}: Global step {loaded_global_step} loaded from checkpoint.")

            except Exception as e:
                if rank == 0:
                    print(f"Rank {rank}: Error loading checkpoint {args.preload_model}: {e}. Initializing model from scratch.")
        else:
            if rank == 0:
                print(f"Rank {rank}: Warning: Preload model path {args.preload_model} does not exist. Initializing model from scratch.")

    model = model.to(DEVICE) # Move the model to the target device
    if rank == 0:
        print(f"Rank {rank}: Model (potentially preloaded) is on device {DEVICE}.")

    if world_size > 1:
        dist.barrier() # Ensure model is on device on all ranks before DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    ema_model = None
    if args.ema_model_saving:
        ema_model = copy.deepcopy(model)
        if rank == 0:
            print("Initialized EMA model")

    psnr_metric = None
    ssim_metric = None
    if (
        rank == 0
        and args.eval_partial_metrics_per_epoch
        and PeakSignalNoiseRatio is not None
    ):
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        print("Initialized PSNR and SSIM metrics on Rank 0.")

    optimizer = Adam(model.parameters(), lr=args.lr)

    if loaded_optimizer_state_dict:
        try:
            optimizer.load_state_dict(loaded_optimizer_state_dict)
            # Ensure optimizer state is on the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
            if rank == 0:
                print(f"Rank {rank}: Optimizer state loaded and moved to device {DEVICE}.")
        except Exception as e:
            if rank == 0:
                print(f"Rank {rank}: Error loading optimizer state: {e}. Optimizer initialized fresh.")

    scheduler = None
    if args.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.lr_plateau_min_lr,
            threshold=args.lr_plateau_threshold,
            verbose=(rank == 0),
        )
        if rank == 0:
            print(f"Initialized ReduceLROnPlateau scheduler.")
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_plateau_min_lr
        )
        if rank == 0:
            print("Initialized CosineAnnealingLR scheduler")
    elif args.scheduler_type != "none" and rank == 0:
        print(f"Error: Invalid scheduler type: {args.scheduler_type}")
        if world_size > 1:
            dist.barrier()
        sys.exit(1)

    if scheduler and loaded_scheduler_state_dict:
        try:
            scheduler.load_state_dict(loaded_scheduler_state_dict)
            if rank == 0:
                print(f"Rank {rank}: Scheduler state loaded.")
        except Exception as e:
            if rank == 0:
                print(f"Rank {rank}: Error loading scheduler state: {e}. Scheduler initialized fresh.")

    early_stopping = None
    if args.early_stopping and len(full_val_split) > 0:
        if rank == 0:  # Early stopping is managed by rank 0
            metric_direction = (
                "minimize" if args.early_stopping_metric == "val_loss" else "maximize"
            )
            early_stopping_path = os.path.join(
                checkpoints_folder, "early_stopping_best_model.pt"
            )
            early_stopping = EarlyStopping(
                patience=args.early_stopping_patience,
                verbose=args.early_stopping_verbose,
                delta=args.early_stopping_delta,
                path=early_stopping_path,
                metric_direction=metric_direction,
                save_scheduler=scheduler,
            )
            print(
                f"Initialized early stopping on Rank 0: metric={args.early_stopping_metric}, direction={metric_direction}"
            )
    elif args.early_stopping and rank == 0:
        print(
            "Warning (Rank 0): Early stopping enabled but validation set is empty. Disabling."
        )
        args.early_stopping = False

    betas = get_beta_schedule(
        args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=args.num_timesteps,
    )
    alphas, alphas_cumprod_np = get_alpha_schedule(betas)
    alphas_cumprod = torch.from_numpy(alphas_cumprod_np).float().to(DEVICE)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    if args.enable_wandb and rank == 0:
        wandb.init(
            project="dl_deblur_ddp",
            config={
                k: str(v) if isinstance(v, (tuple, list)) else v
                for k, v in vars(args).items()
            },
            name=RUN_ID,
            group=args.run_string,
        )
        wandb.watch(
            model, log="gradients", log_freq=1000 * args.gradient_accumulation_steps
        )  # Log less frequently with DDP

    global_step = 0
    if loaded_global_step is not None:
        global_step = loaded_global_step
        if rank == 0:
            print(f"Rank {rank}: Initialized global_step to {global_step} from checkpoint.")

    best_val_loss_rank0 = float("inf")  # Tracked on rank 0

    if rank == 0:
        print("Starting training...")
    for epoch in range(args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch) 

        model.train()
        if ema_model is not None:
            ema_model.train() # Keep EMA model in train mode for updates

        total_train_loss_sum_all_gpus = torch.tensor(
            0.0, device=DEVICE
        )  # For summing losses across GPUs
        num_train_samples_processed_all_gpus = torch.tensor(0, device=DEVICE)

        if rank == 0:
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{args.epochs} (Rank 0)",
                leave=True,
            )

        for batch_idx, batch in enumerate(train_loader):
            blurred_images, sharp_images = batch
            blurred_images = blurred_images.to(DEVICE)
            sharp_images = sharp_images.to(DEVICE)
            current_batch_size_gpu = sharp_images.shape[
                0
            ]  # Samples processed by this GPU in this step

            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True) 

            t = torch.randint(
                0, args.num_timesteps, (current_batch_size_gpu,), device=DEVICE
            ).long()
            noise = torch.randn_like(sharp_images)

            sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alpha_t * sharp_images + sqrt_one_minus_alpha_t * noise

            model_input = torch.cat([x_t, blurred_images], dim=1)
            predicted_noise = model(x=model_input, t=t)

            loss = F.mse_loss(noise, predicted_noise)
            # DDP handles averaging gradients, so no need to divide loss by world_size here
            # before backward if reduction is 'mean' (default).
            # For gradient accumulation, we scale the loss.
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if args.ema_model_saving and ema_model is not None:
                    ema(model, ema_model, args.ema_model_saving_decay)

            # Accumulate loss for averaging at epoch end
            # We use loss.item() which is already averaged over the batch on this GPU
            total_train_loss_sum_all_gpus += loss.item() * current_batch_size_gpu
            num_train_samples_processed_all_gpus += current_batch_size_gpu

            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.item()
                )  # Show loss for rank 0's current batch
                if (
                    args.enable_wandb
                    and (batch_idx + 1) % args.gradient_accumulation_steps == 0
                ):  # Log after optimizer step
                    wandb.log(
                        {"train_loss_step": loss.item()},
                        step=global_step * world_size + batch_idx,
                    )  # Approx global step

        if rank == 0:
            progress_bar.close()

        # Synchronize and average training loss
        if world_size > 1:
            dist.all_reduce(total_train_loss_sum_all_gpus, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_train_samples_processed_all_gpus, op=dist.ReduceOp.SUM)

        avg_train_loss = 0.0
        if num_train_samples_processed_all_gpus.item() > 0:
            avg_train_loss = (
                total_train_loss_sum_all_gpus.item()
                / num_train_samples_processed_all_gpus.item()
            )
        else:  # Should not happen if train_loader has data
            if rank == 0:
                print("Warning: No training samples processed in epoch.")

        global_step += len(train_loader)  # Number of optimizer steps per epoch (approx)

        # - Validation -
        model.eval()
        if ema_model is not None:
            ema_model.eval()

        total_val_loss_sum_all_gpus = torch.tensor(0.0, device=DEVICE)
        num_val_samples_processed_all_gpus = torch.tensor(0, device=DEVICE)

        if rank == 0:
            val_progress_bar = tqdm(
                total=len(val_loader), desc="Validation (Rank 0)", leave=True
            )

        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(val_loader):
                blurred_images_val, sharp_images_val = batch_val
                blurred_images_val = blurred_images_val.to(DEVICE)
                sharp_images_val = sharp_images_val.to(DEVICE)
                current_batch_size_val_gpu = sharp_images_val.shape[0]

                t_val = torch.randint(
                    0, args.num_timesteps, (current_batch_size_val_gpu,), device=DEVICE
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

                model_input_val = torch.cat([x_t_val, blurred_images_val], dim=1)

                # If DDP, model is DDP wrapped. If not, it's the raw model.
                predicted_noise_val = model(x=model_input_val, t=t_val)
                val_loss_item = F.mse_loss(noise_val, predicted_noise_val)

                total_val_loss_sum_all_gpus += (
                    val_loss_item.item() * current_batch_size_val_gpu
                )
                num_val_samples_processed_all_gpus += current_batch_size_val_gpu

                if rank == 0:
                    val_progress_bar.update(1)

        if rank == 0:
            val_progress_bar.close()

        # Synchronize and average validation loss
        if world_size > 1:
            dist.all_reduce(total_val_loss_sum_all_gpus, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_val_samples_processed_all_gpus, op=dist.ReduceOp.SUM)

        avg_val_loss = 0.0
        if num_val_samples_processed_all_gpus.item() > 0:
            avg_val_loss = (
                total_val_loss_sum_all_gpus.item()
                / num_val_samples_processed_all_gpus.item()
            )
        elif rank == 0:  # Only print warning on rank 0
            print(
                "Warning: Validation set is empty or no samples processed during validation."
            )

        # - Rank 0: Logging, Sampling, Metrics, Checkpointing, Early Stopping -
        val_psnr_accum_rank0 = None
        val_ssim_accum_rank0 = None

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} (Rank 0) - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            if (
                args.eval_partial_metrics_per_epoch
                and fixed_val_blurred is not None
                and actual_val_samples > 0
            ):
                print(
                    f"Rank 0: Generating {actual_val_samples} validation samples via DDIM (batches of {val_batch_size})..."
                )
                all_generated_samples = []

                # Use EMA model for inference if enabled and available, otherwise main model (DDP unwrapped)
                inference_model_rank0 = (
                    ema_model.module
                    if args.ema_model_saving and ema_model is not None
                    else model.module
                )
                inference_model_rank0.eval() # Ensure it's in eval mode

                val_sampler_ddim = (
                    DDIMSampler( # Initialize on rank 0 with rank 0's model
                        inference_model_rank0,
                        beta_schedule=args.beta_schedule,
                        beta_start=args.beta_start,
                        beta_end=args.beta_end,
                        num_diffusion_timesteps=args.num_timesteps,
                        device=DEVICE,
                    )
                )

                num_sampling_batches = (
                    actual_val_samples + val_batch_size - 1
                ) // val_batch_size
                for i in tqdm(
                    range(num_sampling_batches),
                    desc="DDIM Sampling (Rank 0)",
                    leave=False,
                ):
                    start_idx = i * val_batch_size
                    end_idx = min((i + 1) * val_batch_size, actual_val_samples)
                    current_sampling_batch_size = end_idx - start_idx
                    if current_sampling_batch_size <= 0:
                        continue

                    condition_batch = fixed_val_blurred[start_idx:end_idx]
                    generated_batch = val_sampler_ddim.sample(
                        batch_size=current_sampling_batch_size,
                        image_size=(C, args.img_height, args.img_width),
                        num_inference_steps=args.val_num_inference_steps,
                        eta=args.val_eta,
                        condition=condition_batch,
                    )
                    all_generated_samples.append(
                        generated_batch.cpu()
                    ) # Move to CPU to save GPU mem

                generated_samples_neg1_1 = torch.cat(all_generated_samples, dim=0).to(
                    DEVICE
                ) # Back to device for metrics

                if psnr_metric is not None and ssim_metric is not None:
                    generated_01 = (generated_samples_neg1_1.clamp(-1, 1) + 1) / 2.0
                    sharp_gt_01 = (fixed_val_sharp.clamp(-1, 1) + 1) / 2.0

                    val_psnr_accum_rank0 = psnr_metric(generated_01, sharp_gt_01).item()
                    val_ssim_accum_rank0 = ssim_metric(generated_01, sharp_gt_01).item()
                    print(
                        f"[Metrics Rank 0] Epoch {epoch+1}: PSNR = {val_psnr_accum_rank0:.4f}, SSIM = {val_ssim_accum_rank0:.4f}"
                    )

                save_val_samples(
                    epoch,
                    fixed_val_blurred,
                    fixed_val_sharp,
                    generated_samples_neg1_1,
                    samples_folder,
                    rank,
                )

            log_dict = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if val_psnr_accum_rank0 is not None:
                log_dict["val_psnr"] = val_psnr_accum_rank0
            if val_ssim_accum_rank0 is not None:
                log_dict["val_ssim"] = val_ssim_accum_rank0

            if args.enable_wandb:
                wandb.log(log_dict, step=global_step) # Use accumulated global_step

            if scheduler is not None:
                if args.scheduler_type == "plateau":
                    scheduler.step(avg_val_loss)
                elif args.scheduler_type == "cosine":
                    scheduler.step()

            if args.early_stopping and early_stopping is not None:
                metric_for_early_stop = None
                if args.early_stopping_metric == "val_loss":
                    metric_for_early_stop = avg_val_loss
                elif (
                    args.early_stopping_metric == "psnr"
                    and val_psnr_accum_rank0 is not None
                ):
                    metric_for_early_stop = val_psnr_accum_rank0
                elif (
                    args.early_stopping_metric == "ssim"
                    and val_ssim_accum_rank0 is not None
                ):
                    metric_for_early_stop = val_ssim_accum_rank0

                if metric_for_early_stop is not None:
                    model_to_save_for_early_stop = (
                        ema_model.module
                        if args.ema_model_saving and ema_model
                        else model.module
                    )
                    early_stopping(
                        metric_for_early_stop,
                        model_to_save_for_early_stop,
                        optimizer,
                        epoch,
                        global_step,
                    )
                    if early_stopping.early_stop:
                        print(
                            f"Rank 0: Early stopping triggered based on {args.early_stopping_metric}"
                        )
                        # Broadcast early stopping signal to other processes
                        stop_signal = torch.tensor(1, device=DEVICE)
                    else:
                        stop_signal = torch.tensor(
                            0, device=DEVICE
                        ) # Ensure stop_signal is defined if not stopping yet
                else:
                    print(
                        f"Warning (Rank 0): Could not compute {args.early_stopping_metric} for early stopping"
                    )
                    stop_signal = torch.tensor(0, device=DEVICE)  # No stop
            else:  # No early stopping or not rank 0
                stop_signal = torch.tensor(0, device=DEVICE)  # No stop

        # End of epoch: check for early stopping signal from rank 0
        if world_size > 1:
            if rank != 0:  # Initialize stop_signal for non-rank 0 processes
                stop_signal = torch.tensor(0, device=DEVICE)
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1:
                if rank != 0:
                    print(f"Rank {rank}: Received early stopping signal from rank 0.")
                break # Break training loop on all processes
        elif (
            rank == 0
            and args.early_stopping
            and early_stopping is not None
            and early_stopping.early_stop
        ):
            break # For single GPU case with early stopping

        if world_size > 1:
            dist.barrier() # Synchronize all processes before starting next epoch or exiting

    if rank == 0:
        print("Training Finished.")
        print(f"Best validation loss on Rank 0: {best_val_loss_rank0:.4f}")
        if args.enable_wandb:
            wandb.finish()

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    main()
