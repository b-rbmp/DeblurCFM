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

from src.deblur.cfm.cfm import ConditionalFlowMatcher
from torchdiffeq import (
    odeint_adjoint as odeint,
) 

from src.models.unet import UNetModel
from src.deblur.datasets.deblur import make_deblur_splits
from src.utils.early_stopping import EarlyStopping
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

# - DDP Imports -
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# - DDP Setup and Cleanup -
def setup_ddp(rank, world_size, local_rank):
    """Initializes the distributed environment."""
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method="env://"
    )
    torch.cuda.set_device(local_rank)
    dist.barrier()
    if rank == 0:
        print(
            f"DDP Initialized: Rank {rank}/{world_size} on GPU {torch.cuda.current_device()} (Local Rank: {local_rank})"
        )


def cleanup_ddp():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        if int(os.environ.get("RANK", 0)) == 0:  # Print only on global rank 0
            print("Cleaned up DDP.")


# - Argument Parsing -
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Conditional Flow Matching for Image Deblurring (DDP)"
    )
    # - Data/Paths -
    parser.add_argument(
        "--blurred_dir", type=str, default="./data/output_text_deblur/blurred"
    )
    parser.add_argument(
        "--sharp_dir", type=str, default="./data/output_text_deblur/sharp"
    )
    parser.add_argument("--img_height", type=int, default=272)
    parser.add_argument("--img_width", type=int, default=480)
    parser.add_argument("--run_string", type=str, default="deblur_cfm_ddp")
    parser.add_argument(
        "--enable_wandb", action=argparse.BooleanOptionalAction, default=False
    )

    # - Training -
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-GPU training batch size"
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation sampling batch size (on rank 0)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers per DataLoader per GPU",
    )
    parser.add_argument("--grad_clip_val", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)

    # - Scheduler -
    parser.add_argument("--scheduler_type", type=str, default="plateau")
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument("--lr_plateau_patience", type=int, default=5)
    parser.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_plateau_threshold", type=float, default=1e-4)

    # - CFM Specific -
    parser.add_argument("--flow_matching_method", type=str, default="vanilla")
    parser.add_argument("--cfm_sigma", type=float, default=0.01)

    # - ODE Solver Specific -
    parser.add_argument("--ode_solver", type=str, default="rk4")
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--ode_atol", type=float, default=1e-3)
    parser.add_argument("--ode_rtol", type=float, default=1e-3)

    # - Model Specific Args (UNet) -
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", type=str, default="1,2,2,2")
    parser.add_argument("--attention_resolutions", type=str, default="16")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # - Validation Sampling -
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=20,
        help="Number of images for validation sampling (on rank 0)",
    )

    # - EMA Model Saving -
    parser.add_argument(
        "--ema_model_saving", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--ema_model_saving_decay", type=float, default=0.9999)

    # - Early Stopping -
    parser.add_argument(
        "--early_stopping", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument(
        "--early_stopping_verbose", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--early_stopping_delta", type=float, default=0.0)
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "psnr", "ssim"],
    )

    # - Partial Metrics Evaluation -
    parser.add_argument(
        "--eval_partial_metrics_per_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # - Model Preloading -
    parser.add_argument(
        "--preload_model",
        type=str,
        default=None,
        help="Path to a .pt model checkpoint to preload (e.g., one saved by early stopping or other training runs).",
    )

    # - DDP Argument -
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for DDP, set by launch script (e.g., torchrun)",
    )

    return parser.parse_args()


# - Helper to save validation images (Rank 0 only) -
def save_val_samples_ddp(epoch, blurred_cond, sharp_gt, generated, sample_dir, rank):
    if rank != 0:
        return
    epoch_dir = os.path.join(sample_dir, f"epoch_{epoch+1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    blurred_cond, sharp_gt, generated = map(
        lambda x: x.cpu().detach(), [blurred_cond, sharp_gt, generated]
    )
    blurred_cond, sharp_gt, generated = map(
        lambda x: (x.clamp(-1, 1) + 1) / 2.0, [blurred_cond, sharp_gt, generated]
    )

    num_samples = generated.shape[0]
    for i in range(num_samples):
        comparison_grid = torch.cat([blurred_cond[i], generated[i], sharp_gt[i]], dim=2)
        vutils.save_image(
            comparison_grid, os.path.join(epoch_dir, f"val_sample_{i:02d}.png")
        )
    print(f"Rank {rank}: Saved {num_samples} validation samples to {epoch_dir}")


# - ODE Dynamics Wrapper -
class FlowDynamics(nn.Module):
    def __init__(
        self, model_or_module: nn.Module, condition: torch.Tensor
    ):
        super().__init__()
        self.model_module = (
            model_or_module
        )
        self.condition = condition

    def forward(self, t, x):
        batch_size = x.shape[0]
        if isinstance(t, float):
            t_input = torch.full((batch_size,), t, device=x.device, dtype=x.dtype)
        elif t.ndim == 0:
            t_input = t.expand(batch_size)
        else:
            t_input = t

        current_condition = self.condition
        if self.condition.shape[0] != batch_size:
            if (
                batch_size == 1 and self.condition.shape[0] > 1
            ):  # Sample one from condition
                current_condition = self.condition[0:1]
            elif (
                self.condition.shape[0] == 1 and batch_size > 1
            ):  # Repeat condition for batch
                current_condition = self.condition.expand(batch_size, -1, -1, -1)
            else:
                print(
                    f"Warning: ODE batch size {batch_size} vs condition {self.condition.shape[0]}. Using direct condition."
                )

        model_input = torch.cat([x, current_condition], dim=1)
        vector_field = self.model_module(
            x=model_input, t=t_input
        )  # Use the unwrapped model
        return vector_field


def ema_ddp(model, ema_model, decay):
    """Update the EMA model weights using exponential moving average for DDP."""
    # Access underlying model if DDP wrapped
    model_params = (
        model.module.parameters() if isinstance(model, DDP) else model.parameters()
    )
    ema_model_params = (
        ema_model.parameters()
    )
    with torch.no_grad():
        for param, ema_param in zip(model_params, ema_model_params):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# - Main Training Function -
def main():
    args = parse_args()

    # - DDP Setup -
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))  # Global rank
    local_rank = int(
        os.environ.get("LOCAL_RANK", rank)
    )  # Use global rank as local if LOCAL_RANK not set (for single GPU fallback)

    if world_size > 1:
        setup_ddp(rank, world_size, local_rank)  # Pass local_rank
        DEVICE = torch.device(f"cuda:{local_rank}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Using device: {DEVICE} (Rank: {rank}, Local Rank: {local_rank})")

    val_batch_size = (
        args.val_batch_size if args.val_batch_size is not None else args.batch_size
    )
    if rank == 0:
        print(f"Using validation sampling batch size (on rank 0): {val_batch_size}")

    # - Setup Directories (Rank 0 only) -
    run_artifacts_folder, checkpoints_folder, samples_folder, RUN_ID = (
        None,
        None,
        None,
        None,
    )
    if rank == 0:
        FOLDER = "src/deblur"
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
        print(f"Rank {rank}: Run artifacts will be saved in: {run_artifacts_folder}")

    if world_size > 1:
        run_id_container = [RUN_ID] if rank == 0 else [None]
        dist.broadcast_object_list(run_id_container, src=0)
        if rank != 0:
            RUN_ID = run_id_container[0]
        dist.barrier()

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
        full_train_split, full_val_split = make_deblur_splits(
            args.blurred_dir, args.sharp_dir, transform, val_ratio=0.2
        )
    except Exception as e:
        if rank == 0:
            print(f"Error creating dataset splits: {e}")
        if world_size > 1:
            dist.barrier()
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
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(world_size > 1),
    )
    val_loader = DataLoader(
        full_val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    if rank == 0:
        print(
            f"Train samples: {len(full_train_split)}, Val samples: {len(full_val_split)}"
        )
        print(
            f"Train batches per GPU: {len(train_loader)}, Val batches per GPU: {len(val_loader)}"
        )

    fixed_val_blurred, fixed_val_sharp, actual_val_samples = None, None, 0
    if rank == 0 and len(full_val_split) > 0:
        num_to_collect = min(args.val_num_samples, len(full_val_split))
        if num_to_collect > 0:
            collected_blurred, collected_sharp = [], []
            temp_val_loader_rank0 = DataLoader(
                full_val_split, batch_size=val_batch_size, shuffle=False
            )
            total_collected = 0
            for batch in temp_val_loader_rank0:
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
                        f"Rank {rank} Warning: Collected only {actual_val_samples} samples for fixed validation."
                    )
                print(
                    f"Rank {rank}: Using {actual_val_samples} samples from validation set for visualization."
                )
            else:
                print(
                    f"Rank {rank} Warning: Could not retrieve fixed validation samples."
                )
        else:
            print(f"Rank {rank} Warning: Requested 0 validation samples.")
    elif rank == 0:
        print(
            f"Rank {rank} Warning: Validation set is empty. Skipping fixed sampling and metrics."
        )
        args.eval_partial_metrics_per_epoch = False

    # - Model -
    model_input_channels = 6
    if rank == 0:
        print(f"Configuring UNet with input channels = {model_input_channels}")
    channel_mult = tuple(map(int, args.channel_mult.split(",")))

    # Parse attention_resolutions (all ranks, or parse on rank 0 and broadcast)
    attention_resolutions_tuple = tuple()
    try:
        if args.attention_resolutions.strip():
            attention_resolutions_list = [
                int(ds_factor.strip())
                for ds_factor in args.attention_resolutions.split(",")
                if ds_factor.strip()
            ]
            if rank == 0:
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
        if rank == 0:
            print(
                f"Using attention at downsample factors (ds): {attention_resolutions_tuple}"
            )
    except ValueError as e:
        if rank == 0:
            print(
                f"Error: Invalid attention resolutions format '{args.attention_resolutions}'. Error: {e}"
            )
        if world_size > 1:
            dist.barrier()
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
        use_fp16=args.fp16,
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

                model.load_state_dict(checkpoint['model_state_dict']) # Load state to CPU model
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

    model = model.to(DEVICE) # Now move the model to the target device
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
        base_model_for_ema = model.module if isinstance(model, DDP) else model
        ema_model = copy.deepcopy(base_model_for_ema).to(
            DEVICE
        )
        ema_model.eval()
        if rank == 0:
            print("Initialized EMA model (not DDP wrapped).")

    flow_matcher = ConditionalFlowMatcher(sigma=args.cfm_sigma)  # Same for all ranks
    if rank == 0:
        print(
            f"Using Flow Matcher: {args.flow_matching_method} with sigma={args.cfm_sigma}"
        )

    optimizer = Adam(
        model.parameters(), lr=args.lr
    )

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

    if scheduler and loaded_scheduler_state_dict:
        try:
            scheduler.load_state_dict(loaded_scheduler_state_dict)
            if rank == 0:
                print(f"Rank {rank}: Scheduler state loaded.")
        except Exception as e:
            if rank == 0:
                print(f"Rank {rank}: Error loading scheduler state: {e}. Scheduler initialized fresh.")

    criterion = nn.MSELoss(reduction="none")  # Calculate per-sample loss, then average
    scaler = torch.amp.GradScaler(enabled=args.fp16)  # Each rank has its own scaler

    early_stopping = None
    if args.early_stopping and len(full_val_split) > 0:
        if rank == 0:  # Managed by rank 0
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
                f"Rank {rank}: Initialized early stopping: metric={args.early_stopping_metric}, direction={metric_direction}"
            )
    elif args.early_stopping and rank == 0:
        print(
            f"Rank {rank} Warning: Early stopping enabled but validation set is empty. Disabling."
        )
        args.early_stopping = False

    psnr_metric, ssim_metric = None, None
    if (
        rank == 0
        and args.eval_partial_metrics_per_epoch
        and PeakSignalNoiseRatio is not None
    ):  # Rank 0 only for these metrics objects
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        print(f"Rank {rank}: Initialized PSNR and SSIM metrics.")

    if args.enable_wandb and rank == 0:
        try:
            wandb.init(
                project="deblur_cfm_ddp",
                config=vars(args),
                name=RUN_ID,
                group=args.run_string,
            )
            wandb.watch(
                model,
                log="gradients",
                log_freq=100 * args.gradient_accumulation_steps,
                log_graph=False,
            )
        except Exception as e:
            print(f"Rank {rank}: Could not initialize WandB: {e}. Disabling WandB.")
            args.enable_wandb = False

    global_step = 0
    if loaded_global_step is not None:
        global_step = loaded_global_step
        if rank == 0:
            print(f"Rank {rank}: Initialized global_step to {global_step} from checkpoint.")

    best_val_loss_rank0 = float("inf")
    if rank == 0:
        print("Starting training...")

    for epoch in range(args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        model.train()

        total_train_loss_sum_all_gpus = torch.tensor(0.0, device=DEVICE)
        num_train_samples_processed_all_gpus = torch.tensor(0, device=DEVICE)

        if rank == 0:
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{args.epochs} (Rank 0)",
                leave=True,
            )

        for batch_idx, batch in enumerate(train_loader):
            blurred_images, sharp_images = batch
            x_cond = blurred_images.to(DEVICE)
            x1 = sharp_images.to(DEVICE)
            current_batch_size_gpu = x1.shape[0]

            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            x0_noise = torch.randn_like(x1)
            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16 if args.fp16 else torch.float32,
                enabled=args.fp16,
            ):
                t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                    x0=x0_noise, x1=x1
                )
                model_input = torch.cat([x_t, x_cond], dim=1)
                v_t_pred = model(
                    x=model_input, t=t
                )
                loss_per_sample = criterion(
                    v_t_pred, u_t
                )
                loss = (
                    loss_per_sample.mean() / args.gradient_accumulation_steps
                )

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if args.grad_clip_val is not None and args.grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_val
                    )
                scaler.step(optimizer)
                scaler.update()
                if (
                    args.ema_model_saving and ema_model is not None
                ): 
                    ema_ddp(model, ema_model, args.ema_model_saving_decay)

            # Accumulate loss for averaging at epoch end
            loss_item_unscaled = loss.item() * args.gradient_accumulation_steps
            total_train_loss_sum_all_gpus += (
                loss_item_unscaled * current_batch_size_gpu
            )  # Sum of batch losses
            num_train_samples_processed_all_gpus += current_batch_size_gpu

            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss_item_unscaled:.4f}")

        global_step_increment = (
            len(train_loader) + args.gradient_accumulation_steps - 1
        ) // args.gradient_accumulation_steps
        global_step += global_step_increment  # Tracks optimizer steps

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
        elif rank == 0:
            print(f"Rank {rank} Warning: No training samples processed in epoch.")

        # - Validation Phase -
        avg_val_loss = 0.0
        val_psnr_accum_rank0, val_ssim_accum_rank0 = None, None

        if len(val_loader.dataset) > 0:
            model.eval()  # Main model to eval for DDP
            if ema_model is not None:
                ema_model.eval()  # EMA model always eval

            total_val_loss_sum_all_gpus = torch.tensor(0.0, device=DEVICE)
            num_val_samples_processed_all_gpus = torch.tensor(0, device=DEVICE)

            if rank == 0:
                val_progress_bar = tqdm(
                    total=len(val_loader), desc="Validation Loss (Rank 0)", leave=True
                )

            with torch.no_grad():
                for batch_val in val_loader:
                    blurred_images_val, sharp_images_val = batch_val
                    x_cond_val, x1_val = blurred_images_val.to(
                        DEVICE
                    ), sharp_images_val.to(DEVICE)
                    current_batch_size_val_gpu = x1_val.shape[0]
                    x0_noise_val = torch.randn_like(x1_val)

                    with torch.amp.autocast(
                        device_type=DEVICE.type,
                        dtype=torch.float16 if args.fp16 else torch.float32,
                        enabled=args.fp16,
                    ):
                        t_val, x_t_val, u_t_val = (
                            flow_matcher.sample_location_and_conditional_flow(
                                x0=x0_noise_val, x1=x1_val
                            )
                        )
                        model_input_val = torch.cat([x_t_val, x_cond_val], dim=1)

                        # Use EMA model for validation loss if available, otherwise main DDP model
                        model_for_val_loss = (
                            ema_model
                            if args.ema_model_saving and ema_model is not None
                            else model
                        )
                        v_t_val_pred = model_for_val_loss(x=model_input_val, t=t_val)
                        val_loss_item = criterion(v_t_val_pred, u_t_val).mean()

                    total_val_loss_sum_all_gpus += (
                        val_loss_item.item() * current_batch_size_val_gpu
                    )
                    num_val_samples_processed_all_gpus += current_batch_size_val_gpu
                    if rank == 0:
                        val_progress_bar.update(1)

            if rank == 0:
                val_progress_bar.close()

            if world_size > 1:
                dist.all_reduce(total_val_loss_sum_all_gpus, op=dist.ReduceOp.SUM)
                dist.all_reduce(
                    num_val_samples_processed_all_gpus, op=dist.ReduceOp.SUM
                )

            if num_val_samples_processed_all_gpus.item() > 0:
                avg_val_loss = (
                    total_val_loss_sum_all_gpus.item()
                    / num_val_samples_processed_all_gpus.item()
                )
            elif rank == 0:
                print(f"Rank {rank} Warning: No validation samples processed.")

            # - Rank 0: ODE Sampling, Metrics, Logging, Checkpointing, Early Stopping -
            if rank == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs} (Rank 0) Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )
                if (
                    args.eval_partial_metrics_per_epoch
                    and fixed_val_blurred is not None
                    and actual_val_samples > 0
                ):
                    print(
                        f"Rank {rank}: Generating {actual_val_samples} val samples via ODE (batches of {val_batch_size})..."
                    )
                    all_generated_samples = []
                    ode_time_span = torch.tensor([0.0, 1.0], device=DEVICE)
                    ode_options = (
                        {"step_size": 1.0 / args.ode_steps}
                        if args.ode_solver in ["euler", "rk4", "midpoint"]
                        else None
                    )

                    # For ODE, use the non-DDP wrapped model (ema or unwrapped main model)
                    inference_model_rank0 = (
                        ema_model
                        if args.ema_model_saving and ema_model is not None
                        else (model.module if isinstance(model, DDP) else model)
                    )
                    inference_model_rank0.eval()

                    num_sampling_batches = (
                        actual_val_samples + val_batch_size - 1
                    ) // val_batch_size
                    for i in tqdm(
                        range(num_sampling_batches),
                        desc="ODE Sampling (Rank 0)",
                        leave=False,
                    ):
                        start_idx, end_idx = i * val_batch_size, min(
                            (i + 1) * val_batch_size, actual_val_samples
                        )
                        current_sampling_batch_size = end_idx - start_idx
                        if current_sampling_batch_size <= 0:
                            continue
                        condition_batch = fixed_val_blurred[start_idx:end_idx]
                        x0_noise_batch = torch.randn(
                            (
                                current_sampling_batch_size,
                                C,
                                args.img_height,
                                args.img_width,
                            ),
                            device=DEVICE,
                        )

                        flow_dynamics_func = FlowDynamics(
                            model_or_module=inference_model_rank0,
                            condition=condition_batch,
                        )

                        solution = odeint(
                            flow_dynamics_func,
                            x0_noise_batch,
                            ode_time_span,
                            method=args.ode_solver,
                            options=ode_options,
                            atol=args.ode_atol,
                            rtol=args.ode_rtol,
                        )
                        generated_batch = solution[-1]
                        all_generated_samples.append(
                            generated_batch.cpu()
                        )

                    generated_samples_neg1_1 = torch.cat(
                        all_generated_samples, dim=0
                    ).to(
                        DEVICE
                    )

                    if psnr_metric is not None and ssim_metric is not None:
                        generated_01 = (generated_samples_neg1_1.clamp(-1, 1) + 1) / 2.0
                        sharp_gt_01 = (fixed_val_sharp.clamp(-1, 1) + 1) / 2.0
                        val_psnr_accum_rank0 = psnr_metric(
                            generated_01, sharp_gt_01
                        ).item()
                        val_ssim_accum_rank0 = ssim_metric(
                            generated_01, sharp_gt_01
                        ).item()
                        print(
                            f"[Metrics Rank 0] Epoch {epoch+1}: PSNR = {val_psnr_accum_rank0:.4f}, SSIM = {val_ssim_accum_rank0:.4f}"
                        )

                    save_val_samples_ddp(
                        epoch,
                        fixed_val_blurred,
                        fixed_val_sharp,
                        generated_samples_neg1_1,
                        samples_folder,
                        rank,
                    )

                log_dict_rank0 = {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if val_psnr_accum_rank0 is not None:
                    log_dict_rank0["val_psnr"] = val_psnr_accum_rank0
                if val_ssim_accum_rank0 is not None:
                    log_dict_rank0["val_ssim"] = val_ssim_accum_rank0

                if args.enable_wandb:
                    wandb.log(log_dict_rank0, step=global_step)

                if scheduler is not None:
                    if args.scheduler_type == "plateau":
                        scheduler.step(avg_val_loss)
                    elif args.scheduler_type == "cosine":
                        scheduler.step()

                stop_signal_val = 0
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
                        # For early stopping, save the non-DDP model state
                        model_for_early_stop = (
                            ema_model
                            if args.ema_model_saving and ema_model is not None
                            else (model.module if isinstance(model, DDP) else model)
                        )
                        early_stopping(
                            metric_for_early_stop,
                            model_for_early_stop,
                            optimizer,
                            epoch,
                            global_step,
                        )
                        if early_stopping.early_stop:
                            print(
                                f"Rank {rank}: Early stopping triggered based on {args.early_stopping_metric}"
                            )
                            stop_signal_val = 1
                    else:
                        print(
                            f"Rank {rank} Warning: Could not compute {args.early_stopping_metric} for early stopping."
                        )
        else:
            if rank == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs} (Rank 0) Summary - Train Loss: {avg_train_loss:.4f} (No validation)"
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

        # Broadcast early stopping signal from rank 0
        if world_size > 1:
            stop_signal_tensor = torch.tensor(
                stop_signal_val if rank == 0 and len(val_loader.dataset) > 0 else 0,
                device=DEVICE,
            )
            dist.broadcast(stop_signal_tensor, src=0)
            if stop_signal_tensor.item() == 1:
                if rank != 0:
                    print(f"Rank {rank}: Received early stopping signal.")
                break  # Break training loop on all processes
        elif (
            rank == 0
            and len(val_loader.dataset) > 0
            and args.early_stopping
            and early_stopping is not None
            and early_stopping.early_stop
        ):
            break  # Single GPU early stop

        if world_size > 1:
            dist.barrier()  # Sync all processes before next epoch or exit

    # - End Training Loop -
    if rank == 0:
        print("Training Finished.")
        if len(full_val_split) > 0:
            print(f"Best validation loss achieved on Rank 0: {best_val_loss_rank0:.4f}")
        if args.enable_wandb:
            wandb.finish()

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    main()
