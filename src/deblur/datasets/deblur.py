import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split

class DeblurDataset(Dataset):
    """
    Dataset for image deblurring.
    Reads pairs of (blurred, sharp) images from specified folders.
    Assumes filenames match: 'prefix_ID_blurred.ext' and 'prefix_ID_sharp.ext'.
    """
    def __init__(self, blurred_dir: str, sharp_dir: str, transform=None):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.transform = transform

        self.blurred_files = sorted([
            f for f in os.listdir(blurred_dir)
            if os.path.isfile(os.path.join(blurred_dir, f))
        ])

        self.image_ids = []
        for fname in self.blurred_files:
            parts = fname.split('_')
            if len(parts) >= 2 and parts[-1].startswith("blurred"):
                img_id = parts[-2]
                sharp_fname_base = f"{'_'.join(parts[:-1])}_sharp"
                found_sharp = False
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    sharp_fname = f"{sharp_fname_base}{ext}"
                    sharp_path = os.path.join(sharp_dir, sharp_fname)
                    if os.path.exists(sharp_path):
                        self.image_ids.append({
                            "id": img_id,
                            "blurred": fname,
                            "sharp": sharp_fname
                        })
                        found_sharp = True
                        break
                if not found_sharp:
                    print(f"Warning: No sharp file found for blurred image {fname} with base {sharp_fname_base}")
            else:
                 print(f"Warning: Skipping file with unexpected format: {fname}")


        if not self.image_ids:
            raise RuntimeError(f"No valid image pairs found between {blurred_dir} and {sharp_dir}")

        print(f"Found {len(self.image_ids)} image pairs.")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        item = self.image_ids[idx]
        blurred_path = os.path.join(self.blurred_dir, item["blurred"])
        sharp_path = os.path.join(self.sharp_dir, item["sharp"])

        try:
            blurred_img = Image.open(blurred_path).convert("RGB")
            sharp_img = Image.open(sharp_path).convert("RGB")
        except FileNotFoundError as e:
             print(f"Error loading image pair: {e}")
             if idx > 0:
                 return self.__getitem__(0)
             else:
                 raise IOError(f"Failed to load images: {blurred_path} or {sharp_path}") from e
        except Exception as e:
            print(f"Unexpected error loading image {item['id']}: {e}")
            if idx > 0:
                 return self.__getitem__(0)
            else:
                 raise IOError(f"Unexpected error loading images: {blurred_path} or {sharp_path}") from e


        if self.transform is not None:
            blurred_img_t = self.transform(blurred_img)
            sharp_img_t = self.transform(sharp_img)
        else:
            blurred_img_t = blurred_img
            sharp_img_t = sharp_img


        return blurred_img_t, sharp_img_t

def make_deblur_splits(blurred_dir, sharp_dir, transform, val_ratio=0.1, seed=42):
    """Creates training and validation splits for the DeblurDataset."""
    full_dataset = DeblurDataset(blurred_dir, sharp_dir, transform=transform)

    if not full_dataset:
         raise ValueError("Dataset creation failed, cannot make splits.")

    n_total = len(full_dataset)
    if n_total == 0:
        raise ValueError("Dataset is empty, cannot create splits.")

    n_val = int(n_total * val_ratio)
    n_val = max(0, min(n_val, n_total))
    n_train = n_total - n_val

    if n_train <= 0 or n_val < 0:
         raise ValueError(f"Invalid split sizes: n_train={n_train}, n_val={n_val}. Check val_ratio and dataset size.")


    print(f"Splitting dataset: {n_train} train, {n_val} validation samples.")
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed) # Use fixed seed for reproducibility
    )
    return train_dataset, val_dataset