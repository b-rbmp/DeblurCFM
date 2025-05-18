import sys
import os
import argparse
import datetime
import json

sys.path.append(os.getcwd())  

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.unet import UNetModel 
from src.deblur.datasets.deblur import make_deblur_splits
from torchdiffeq import odeint_adjoint as odeint
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

import wandb

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
        t_input = t.expand(batch_size)
        current_condition = self.condition

        if self.condition.shape[0] != batch_size:

            if self.condition.shape[0] == 1 and batch_size > 1 : # condition needs repeating
                 current_condition = self.condition.expand(batch_size, -1, -1, -1)
            elif self.condition.shape[0] > 1 and batch_size == 1: # ode solver processes one by one
                print(f"Warning: ODE batch size {batch_size} differs from condition batch size {self.condition.shape[0]}. Using first condition for single ODE item.")
                current_condition = self.condition[0:1] # Fallback, may not be universally correct
            else:
                 raise RuntimeError(
                    f"ODE batch size {batch_size} and condition batch size {self.condition.shape[0]} are incompatible."
                )
        model_input = torch.cat([x, current_condition], dim=1)
        vector_field = self.model(x=model_input, t=t_input)
        return vector_field


# - Argument Parsing -
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Conditional Flow Matching for Image Deblurring")
    # --- Paths ---
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./best_models/cfm/2025-05-16_22-44-52_deblur_cfm_ddp_geeky-zucchini-robin/checkpoints/early_stopping_best_model.pt",
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--blurred_dir",
        type=str,
        default="./data/output_text_deblur/blurred",
        help="Directory containing blurred images for validation",
    )
    parser.add_argument(
        "--sharp_dir",
        type=str,
        default="./data/output_text_deblur/sharp",
        help="Directory containing sharp images for validation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results (metrics, images)",
    )
    parser.add_argument(
        "--run_string", type=str, default="deblur_cfm_eval", help="Run String for output subfolder"
    )

    # - Data -
    parser.add_argument("--img_height", type=int, default=272, help="Image height")
    parser.add_argument("--img_width", type=int, default=480, help="Image width")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoader"
    )

    # - ODE Solver Specific -
    parser.add_argument(
        "--ode_solver",
        type=str,
        default="dopri5",
        help="ODE solver for sampling (e.g., 'rk4', 'dopri5', 'euler')",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=10,
        help="Number of steps for ODE solver during sampling for fixed-step solvers",
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

    # - Model Specific Args (UNet) - Must match the trained model -
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
        "--channel_mult", type=str, default="1,2,2,2", help="Channel multipliers (comma-separated)"
    )
    parser.add_argument(
        "--attention_resolutions", type=str, default="16", help="Attention resolutions (comma-separated)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mixed-precision for inference (if model supports/trained with it)",
    )


    # - Evaluation Specific -
    parser.add_argument(
        "--save_images_count",
        type=int,
        default=100,
        help="Number of image triplets (blurred, generated, sharp) to save. Set to 0 to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu'). Auto-detects if None.",
    )
    parser.add_argument(
        "--enable_wandb", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


# - Helper to save evaluation images -
def save_evaluation_samples(base_dir, batch_idx, blurred_cond, sharp_gt, generated, count_to_save):
    os.makedirs(base_dir, exist_ok=True)
    blurred_cond, sharp_gt, generated = map(
        lambda x: x.cpu().detach(), [blurred_cond, sharp_gt, generated]
    )
    # - Clamp and scale images from [-1, 1] to [0, 1] for saving -
    blurred_cond, sharp_gt, generated = map(
        lambda x: (x.clamp(-1, 1) + 1) / 2.0, [blurred_cond, sharp_gt, generated]
    )

    num_in_batch = generated.shape[0]
    for i in range(min(num_in_batch, count_to_save)): # Save up to count_to_save from this batch
        img_idx = batch_idx * blurred_cond.shape[0] + i # Global image index
        comparison_grid = torch.cat([blurred_cond[i], generated[i], sharp_gt[i]], dim=2) # W, 3*H
        vutils.save_image(
            comparison_grid, os.path.join(base_dir, f"eval_sample_{img_idx:04d}.png")
        )

# - Main Evaluation Function -
def evaluate():
    args = parse_args()
    DEVICE = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {DEVICE}")
    RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + "CFM_EVAL"
    if args.enable_wandb:
        wandb.init(
            project="test_deblur",
            config=vars(args),
            name=RUN_ID,
            group=args.run_string,
        )

    # - Setup Output Directory -
    eval_run_name = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_string
    )
    results_dir = os.path.join(args.output_dir, eval_run_name)
    images_save_dir = os.path.join(results_dir, "saved_images")
    os.makedirs(results_dir, exist_ok=True)
    if args.save_images_count > 0:
        os.makedirs(images_save_dir, exist_ok=True)
    print(f"Evaluation results will be saved in: {results_dir}")

    # - Dataset -
    IMG_DIMS = (args.img_height, args.img_width)
    C = 3
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_DIMS),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1), # Scale to [-1, 1]
        ]
    )
    print(f"Loading validation dataset from: Blurred='{args.blurred_dir}', Sharp='{args.sharp_dir}'")
    try:

        _ , val_split = make_deblur_splits( # Assuming it returns (train, val)
             args.blurred_dir, args.sharp_dir, transform, val_ratio=0.2, seed=42 # Same seed to get the same validation set
        )

        if not val_split or len(val_split) == 0:
            print("Error: Validation split is empty. Check data paths and splitting logic.")
            sys.exit(1)

    except Exception as e:
        print(f"Error creating dataset splits: {e}")
        sys.exit(1)

    val_loader = DataLoader(
        val_split,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    print(f"Validation samples: {len(val_split)}, Validation batches: {len(val_loader)}")
    if len(val_split) == 0:
        print("No validation data found. Exiting.")
        return

    # - Load Model -
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)

    # Reconstruct model architecture
    model_input_channels = 6
    channel_mult_tuple = tuple(map(int, args.channel_mult.split(",")))
    try:  # Parse attention resolutions
        attention_ds = []
        image_size_for_attn = args.img_width
        for res_str in args.attention_resolutions.split(","):
            res = int(res_str)
            assert res > 0
            attention_ds.append(max(1, image_size_for_attn // res))
        attention_resolutions_tuple = tuple(attention_ds)
    except Exception as e:
        print(
            f"Error parsing attention resolutions '{args.attention_resolutions}': {e}"
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
        channel_mult=channel_mult_tuple,
        use_checkpoint=False,
        use_fp16=args.fp16,
        num_heads=args.num_heads,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(DEVICE)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded and set to evaluation mode.")

    # - Metrics Initialization -
    # PSNR/SSIM expect inputs in [0, 1] range, set data_range=1.0
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # - Evaluation Loop -
    num_samples_processed = 0
    images_saved_so_far = 0

    ode_time_span = torch.tensor([0.0, 1.0], device=DEVICE) # Integrate from t=0 (noise) to t=1 (image)
    ode_options = None
    if args.ode_solver in ["euler", "rk4", "midpoint"]: # Fixed step solvers
        ode_options = {"step_size": 1.0 / args.ode_steps}

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating"):
            blurred_images, sharp_images = batch
            x_cond = blurred_images.to(DEVICE) # Blurred condition
            x1_gt = sharp_images.to(DEVICE)    # Ground truth sharp
            current_batch_size = x_cond.shape[0]

            # Initial state for ODE is noise x0 ~ N(0, I)
            x0_noise_batch = torch.randn(
                (current_batch_size, C, args.img_height, args.img_width), device=DEVICE
            )

            # - ODE Integration -
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16 if args.fp16 else torch.float32, enabled=args.fp16):
                flow_dynamics_func = FlowDynamics(
                    model=model,
                    condition=x_cond,
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
                generated_samples_neg1_1 = solution[-1] # State at t=1

            # - Calculate PSNR/SSIM for the current batch -
            # Convert generated and GT images from [-1, 1] to [0, 1] for metrics
            generated_01 = (generated_samples_neg1_1.clamp(-1, 1) + 1) / 2.0
            sharp_gt_01 = (x1_gt.clamp(-1, 1) + 1) / 2.0

            psnr_metric.update(generated_01, sharp_gt_01)
            ssim_metric.update(generated_01, sharp_gt_01)
            if args.enable_wandb:
                wandb.log({
                    "psnr": psnr_metric.compute().item(),
                    "ssim": ssim_metric.compute().item(),
                })
            num_samples_processed += current_batch_size

            # - Save some sample images -
            if args.save_images_count > 0 and images_saved_so_far < args.save_images_count:
                can_save_this_batch = args.save_images_count - images_saved_so_far
                save_evaluation_samples(
                    images_save_dir,
                    batch_idx, # Use batch_idx to give unique names
                    x_cond, # Blurred condition [-1, 1]
                    x1_gt, # Ground Truth [-1, 1]
                    generated_samples_neg1_1, # Generated [-1, 1]
                    count_to_save=min(can_save_this_batch, current_batch_size)
                )
                images_saved_so_far += min(can_save_this_batch, current_batch_size)

        

    # - Compute Final Global Metrics -
    global_psnr = psnr_metric.compute().item()
    global_ssim = ssim_metric.compute().item()

    print("\n--- Evaluation Summary ---")
    print(f"Processed {num_samples_processed} images from the validation set.")
    print(f"Global PSNR: {global_psnr:.4f}")
    print(f"Global SSIM: {global_ssim:.4f}")
    if args.save_images_count > 0:
        print(f"Saved {images_saved_so_far} sample images to: {images_save_dir}")

    # - Save Metrics to a File -
    metrics_summary = {
        "checkpoint_path": args.checkpoint_path,
        "dataset_blurred_dir": args.blurred_dir,
        "dataset_sharp_dir": args.sharp_dir,
        "num_samples_processed": num_samples_processed,
        "global_psnr": global_psnr,
        "global_ssim": global_ssim,
        "ode_solver": args.ode_solver,
        "ode_steps": args.ode_steps if args.ode_solver in ["euler", "rk4", "midpoint"] else "N/A (adaptive)",
        "ode_atol": args.ode_atol if args.ode_solver not in ["euler", "rk4", "midpoint"] else "N/A (fixed-step)",
        "ode_rtol": args.ode_rtol if args.ode_solver not in ["euler", "rk4", "midpoint"] else "N/A (fixed-step)",
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
    }
    metrics_file_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics summary saved to: {metrics_file_path}")

    # Save args used for this evaluation run
    args_file_path = os.path.join(results_dir, "evaluation_args.json")
    with open(args_file_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Evaluation arguments saved to: {args_file_path}")

    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    evaluate()