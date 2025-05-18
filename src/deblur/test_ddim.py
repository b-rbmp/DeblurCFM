import sys
import os
import argparse
import datetime
import json

sys.path.append(os.getcwd())  # Add the current working directory to the path

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.unet import UNetModel
from src.deblur.datasets.deblur import make_deblur_splits
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
import wandb


from src.deblur.ddim_sampler import DDIMSampler

# - Argument Parsing -
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDIM for Image Deblurring")
    # - Paths -
    parser.add_argument("--checkpoint_path", type=str, default="./best_models/diff/2025-05-17_12-00-57_deblur_ddp_wiggy-salmon-javanese/checkpoints/early_stopping_best_model.pt", help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--blurred_dir", type=str, default="./data/output_text_deblur/blurred", help="Directory containing blurred images for validation")
    parser.add_argument("--sharp_dir", type=str, default="./data/output_text_deblur/sharp", help="Directory containing sharp images for validation")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results (metrics, images)")
    parser.add_argument("--run_string", type=str, default="deblur_ddim_eval", help="Run String for output subfolder")

    # - Data -
    parser.add_argument("--img_height", type=int, default=272, help="Image height")
    parser.add_argument("--img_width", type=int, default=480, help="Image width")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Evaluation batch size (DDIM sampling can be slow, small batch often preferred)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")

    # - DDIM Sampler Specific -
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps (T) model was trained with")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule type (e.g., linear)")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta schedule start value")
    parser.add_argument("--beta_end", type=float, default=2e-2, help="Beta schedule end value")
    parser.add_argument("--val_num_inference_steps", type=int, default=100, help="Number of DDIM inference steps for sampling")
    parser.add_argument("--val_eta", type=float, default=0.0, help="DDIM eta (0.0 for DDIM, 1.0 for DDPM-like sampling with DDIM schedule)")

    # - Model Specific Args (UNet) -
    parser.add_argument("--num_channels", type=int, default=128, help="Base channel count for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per level")
    parser.add_argument("--channel_mult", type=str, default="1,2,2,2", help="Channel multipliers (comma-separated)")
    parser.add_argument("--attention_resolutions", type=str, default="16", help="Feature map resolutions for attention (comma-separated, e.g., 32,16)")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # - Evaluation Specific -
    parser.add_argument("--save_images_count", type=int, default=100, help="Number of image triplets to save. 0 to disable.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detects if None.")
    parser.add_argument("--fp16_eval", action=argparse.BooleanOptionalAction, default=False, help="Enable AMP for evaluation loop (model forward pass during sampling)")
    parser.add_argument(
        "--enable_wandb", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()

# - Helper to save evaluation images -
def save_evaluation_samples(base_dir, batch_idx, current_batch_size_val, blurred_cond, sharp_gt, generated, count_to_save, global_img_offset):
    os.makedirs(base_dir, exist_ok=True)
    blurred_cond, sharp_gt, generated = map(
        lambda x: x.cpu().detach(), [blurred_cond, sharp_gt, generated]
    )
    blurred_cond, sharp_gt, generated = map(
        lambda x: (x.clamp(-1, 1) + 1) / 2.0, [blurred_cond, sharp_gt, generated] # Scale to [0,1]
    )

    num_in_batch = generated.shape[0]
    for i in range(min(num_in_batch, count_to_save)):
        img_idx = global_img_offset + i
        comparison_grid = torch.cat([blurred_cond[i], generated[i], sharp_gt[i]], dim=2)
        vutils.save_image(
            comparison_grid, os.path.join(base_dir, f"eval_sample_{img_idx:04d}.png")
        )
    return min(num_in_batch, count_to_save)


# - Main Evaluation Function -
def evaluate_ddim():
    args = parse_args()
    DEVICE = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
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
    eval_run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_string
    results_dir = os.path.join(args.output_dir, eval_run_name)
    images_save_dir = os.path.join(results_dir, "saved_images")
    os.makedirs(results_dir, exist_ok=True)
    if args.save_images_count > 0:
        os.makedirs(images_save_dir, exist_ok=True)
    print(f"Evaluation results will be saved in: {results_dir}")

    # - Dataset -
    IMG_DIMS = (args.img_height, args.img_width)
    C = 3
    transform = transforms.Compose([
        transforms.Resize(IMG_DIMS),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1), # Scale to [-1, 1]
    ])
    print(f"Loading validation dataset from: Blurred='{args.blurred_dir}', Sharp='{args.sharp_dir}'")
    try:
        _, val_split = make_deblur_splits(
            args.blurred_dir, args.sharp_dir, transform, val_ratio=0.2, seed=42 # Same seed to get the same validation set
        )
        if not val_split or len(val_split) == 0:
            print("Error: Validation split is empty. Check data paths and splitting logic.")
            sys.exit(1)
    except Exception as e:
        print(f"Error creating dataset splits: {e}")
        sys.exit(1)

    val_loader = DataLoader(
        val_split, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    print(f"Validation samples: {len(val_split)}, Validation batches: {len(val_loader)}")

    # - Load Model -
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)

    model_input_channels = 6    
    channel_mult_tuple = tuple(map(int, args.channel_mult.split(",")))
    try:
        attention_ds = []
        image_size_for_attn = args.img_width
        for res_str in args.attention_resolutions.split(","):
            res = int(res_str)
            if res == 0:
                raise ValueError("Attention resolution cannot be zero")
            ds_val = max(1, image_size_for_attn // res)
            attention_ds.append(ds_val)
        attention_resolutions_tuple = tuple(attention_ds)
        print(
            f"Attention resolutions mapped to downsample factors: {attention_resolutions_tuple}"
        )
    except (ValueError, ZeroDivisionError) as e:
        print(
            f"Error: Invalid attention resolutions format '{args.attention_resolutions}': {e}"
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
        use_fp16=False,
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
    else: # If the checkpoint is just the state_dict
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # - Initialize DDIM Sampler -
    ddim_sampler = DDIMSampler(
        model,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=args.num_timesteps, # T from training
        device=DEVICE
    )
    print(f"DDIM Sampler initialized with {args.val_num_inference_steps} inference steps, eta={args.val_eta}.")

    # - Metrics Initialization -
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE) # Images will be [0,1]
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # - Evaluation Loop -
    images_saved_total = 0
    global_img_offset_for_saving = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating DDIM"):
            blurred_images, sharp_images_gt = batch
            x_cond = blurred_images.to(DEVICE) # Blurred condition
            x_sharp_gt = sharp_images_gt.to(DEVICE) # Ground truth sharp
            current_batch_size_val = x_cond.shape[0]

            # - DDIM Sampling -
            with torch.amp.autocast(device_type=DEVICE, enabled=args.fp16_eval):
                 generated_samples_neg1_1 = ddim_sampler.sample(
                    batch_size=current_batch_size_val,
                    image_size=(C, args.img_height, args.img_width), # (Channels, H, W)
                    num_inference_steps=args.val_num_inference_steps,
                    eta=args.val_eta,
                    condition=x_cond
                )

            # - Calculate PSNR/SSIM for the current batch -
            generated_01 = (generated_samples_neg1_1.clamp(-1, 1) + 1) / 2.0 # to [0,1]
            sharp_gt_01 = (x_sharp_gt.clamp(-1, 1) + 1) / 2.0 # to [0,1]

            psnr_metric.update(generated_01, sharp_gt_01)
            ssim_metric.update(generated_01, sharp_gt_01)
            if args.enable_wandb:
                wandb.log({
                    "psnr": psnr_metric.compute().item(),
                    "ssim": ssim_metric.compute().item(),
                })
            # - Save some sample images -
            if args.save_images_count > 0 and images_saved_total < args.save_images_count:
                can_save_this_batch = args.save_images_count - images_saved_total
                saved_this_iter = save_evaluation_samples(
                    images_save_dir, batch_idx, current_batch_size_val,
                    x_cond, x_sharp_gt, generated_samples_neg1_1, # Pass [-1,1] images
                    count_to_save=min(can_save_this_batch, current_batch_size_val),
                    global_img_offset = global_img_offset_for_saving
                )
                images_saved_total += saved_this_iter
            global_img_offset_for_saving += current_batch_size_val


    # - Compute Final Global Metrics -
    global_psnr = psnr_metric.compute().item()
    global_ssim = ssim_metric.compute().item()

    print("\n--- DDIM Evaluation Summary ---")
    print(f"Processed {len(val_split)} images from the validation set.")
    print(f"Global PSNR: {global_psnr:.4f}")
    print(f"Global SSIM: {global_ssim:.4f}")
    if args.save_images_count > 0:
        print(f"Saved {images_saved_total} sample images to: {images_save_dir}")

    # - Save Metrics to a File -
    metrics_summary = {
        "checkpoint_path": args.checkpoint_path,
        "dataset_blurred_dir": args.blurred_dir,
        "dataset_sharp_dir": args.sharp_dir,
        "num_samples_processed": len(val_split),
        "global_psnr": global_psnr,
        "global_ssim": global_ssim,
        "num_timesteps_train (T)": args.num_timesteps,
        "val_num_inference_steps": args.val_num_inference_steps,
        "val_eta": args.val_eta,
        "beta_schedule_params": f"{args.beta_schedule}, start={args.beta_start}, end={args.beta_end}",
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
    }
    metrics_file_path = os.path.join(results_dir, "evaluation_metrics_ddim.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics summary saved to: {metrics_file_path}")

    args_file_path = os.path.join(results_dir, "evaluation_args_ddim.json")
    with open(args_file_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Evaluation arguments saved to: {args_file_path}")

    if args.enable_wandb:
        wandb.finish()

if __name__ == "__main__":
    evaluate_ddim()