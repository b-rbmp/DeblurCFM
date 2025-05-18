import os
import random
import io
import sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import datetime
import traceback

# Import SciPy for convolution
try:
    import scipy.signal
except ImportError:
    print("Error: SciPy is required for Motion/Defocus blur.")
    print("Please install it using: pip install scipy")
    sys.exit(1)

# - Configuration -

# Input Resources
FONT_DIR = "/home/bernardoribeiro/Documents/GitHub/DL-Project/dataset_generator/fonts"
TEXT_CORPUS = (
    "/home/bernardoribeiro/Documents/GitHub/DL-Project/dataset_generator/corpus.txt"
)
TEXTURE_DIR = (
    "/home/bernardoribeiro/Documents/GitHub/DL-Project/dataset_generator/textures"
)

# Output Directories
OUTPUT_SHARP_DIR = (
    "/home/bernardoribeiro/Documents/GitHub/DL-Project/data/output_text_deblur/sharp"
)
OUTPUT_BLURRED_DIR = (
    "/home/bernardoribeiro/Documents/GitHub/DL-Project/data/output_text_deblur/blurred"
)

# Image Generation Parameters
NUM_SAMPLES = 100000
IMAGE_WIDTH = int(1920 / 4)
IMAGE_HEIGHT = int(1080 / 4)

# Text Rendering Parameters
FONT_SIZE_RANGE_PT = (18, 72)
TEXT_COLOR_RANGE = ((0, 0, 0), (60, 60, 60))
LINES_PER_SAMPLE = (1, 4)
MAX_WORDS_PER_LINE = 3
MAX_RENDER_TEXT_LENGTH = 200

# Background Parameters
BACKGROUND_TYPE_PROB = {"color": 0.4, "noise": 0.4, "texture": 0.2}
BG_COLOR_RANGE = ((200, 200, 200), (255, 255, 255))
BG_NOISE_INTENSITY_RANGE = (5, 30)

# - Degradation Parameters for Deblurring -
# Probabilities for choosing blur type (must sum to 1.0)
BLUR_TYPE_PROB = {'gaussian': 0.4, 'motion': 0.3, 'defocus': 0.3}

# Gaussian Blur
GAUSSIAN_BLUR_RADIUS_RANGE = (1.0, 5.0) # Radius range for Gaussian blur

# Motion Blur
MOTION_BLUR_LENGTH_RANGE = (6, 40) # Length of motion streak in pixels
MOTION_BLUR_ANGLE_RANGE = (0, 360) # Angle of motion in degrees

# Defocus Blur
DEFOCUS_BLUR_RADIUS_RANGE = (2, 10) # Radius of the disk kernel for defocus

# Other degradations applied AFTER blur
NOISE_INTENSITY_RANGE = (2, 10) # Gaussian noise std dev
JPEG_QUALITY_RANGE = (40, 85) # JPEG compression quality

# - Helper Functions -

def get_random_text(filename, min_lines, max_lines):
    """Reads random consecutive lines from the text corpus."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return "No text found in corpus."
        # Ensure enough lines available if max_lines > 1
        max_start_index = len(lines) - max_lines
        if max_start_index < 0: max_start_index = 0 # Handle case where corpus is small

        num_lines = random.randint(min_lines, max_lines)
        # Adjust num_lines if corpus is too small
        num_lines = min(num_lines, len(lines))
        start_index = random.randint(0, len(lines) - num_lines)

        snippet = "".join(lines[start_index : start_index + num_lines]).strip()
        return snippet if snippet else "Default Text" # Ensure not empty
    except FileNotFoundError:
        print(f"Error reading corpus: File not found at {filename}", file=sys.stderr)
        return "Corpus file not found."
    except Exception as e:
        print(f"Error reading text: {e}", file=sys.stderr)
        return "Error reading text."


def get_random_font(font_dir):
    """Selects a random font file path from the specified directory."""
    try:
        fonts = [
            os.path.join(font_dir, f)
            for f in os.listdir(font_dir)
            if f.lower().endswith((".ttf", ".otf"))
        ]
        return random.choice(fonts) if fonts else None
    except FileNotFoundError:
         print(f"Error listing fonts: Directory not found at {font_dir}", file=sys.stderr)
         return None
    except Exception as e:
        print(f"Error listing fonts: {e}", file=sys.stderr)
        return None


def create_background(width, height, bg_type, color_range, noise_range, texture_dir):
    """Creates a background image based on the specified type."""
    # Texture Background
    if bg_type == "texture":
        try:
            textures = [
                os.path.join(texture_dir, f)
                for f in os.listdir(texture_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if textures:
                texture_path = random.choice(textures)
                img = Image.open(texture_path).convert("RGB")
                # Resize texture to fit background exactly
                return img.resize((width, height), Image.Resampling.LANCZOS)
            else:
                bg_type = "color" # Fallback if no textures
        except FileNotFoundError:
            bg_type = "color" # Fallback
        except Exception as e:
            print(f"Warning: Error loading texture '{texture_path}': {e}. Falling back to color.", file=sys.stderr)
            bg_type = "color" # Fallback

    # Noise Background
    if bg_type == "noise":
        base_color = tuple(random.randint(c_min, c_max) for c_min, c_max in zip(*color_range))
        bg_np = np.full((height, width, 3), base_color, dtype=np.uint8)
        intensity = random.randint(*noise_range)
        noise = np.random.randint(-intensity, intensity + 1, (height, width, 3), dtype=np.int16)
        bg_np = np.clip(bg_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(bg_np, 'RGB')

    # Default or 'color' type
    color = tuple(random.randint(c_min, c_max) for c_min, c_max in zip(*color_range))
    return Image.new('RGB', (width, height), color)


# - Kernel Generation Functions -

def create_motion_blur_kernel(length, angle):
    """Creates a 1D line kernel for motion blur."""
    # Ensure kernel length is odd
    kernel_size = int(length)
    if kernel_size % 2 == 0:
        kernel_size += 1

    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # Convert angle to radians
    rad_angle = np.deg2rad(angle)

    # Calculate line coordinates 
    dx = np.cos(rad_angle)
    dy = -np.sin(rad_angle) # Image coordinates

    for i in range(kernel_size):
        x = int(center + (i - center) * dx + 0.5)
        y = int(center + (i - center) * dy + 0.5)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    # Normalize the kernel
    if np.sum(kernel) == 0: # Handle edge case of zero length/invalid line
        kernel[center, center] = 1.0 # Make it an identity kernel (no blur)
    else:
        kernel /= np.sum(kernel)
    return kernel

def create_defocus_blur_kernel(radius):
    """Creates a disk kernel for defocus blur."""
    kernel_size = int(radius * 2) + 1
    if kernel_size % 2 == 0:
        kernel_size +=1
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    y, x = np.ogrid[-center:kernel_size-center, -center:kernel_size-center]
    mask = x*x + y*y <= radius*radius
    kernel[mask] = 1.0

    # Normalize
    if np.sum(kernel) == 0:
         kernel[center, center] = 1.0
    else:
        kernel /= np.sum(kernel)
    return kernel

# - Degradation Function with Multiple Blur Types -
def apply_degradation_for_deblur(sharp_img):
    """
    Applies a randomly chosen blur (Gaussian, Motion, Defocus),
    then noise, and JPEG compression AT THE SAME RESOLUTION.
    """
    img_pil = sharp_img.copy() # Work on a copy

    # 1. Randomly Choose and Apply Blur
    blur_choice = random.random()
    cumulative_prob = 0
    applied_blur_type = 'none'

    for blur_type, prob in BLUR_TYPE_PROB.items():
        cumulative_prob += prob
        if blur_choice < cumulative_prob:
            applied_blur_type = blur_type
            break

    blurred_img_pil = img_pil # Initialize in case no blur applied

    if applied_blur_type == 'gaussian':
        blur_radius = random.uniform(*GAUSSIAN_BLUR_RADIUS_RANGE)
        blurred_img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    elif applied_blur_type in ['motion', 'defocus']:
        # Convert PIL image to NumPy array (float32 for convolution)
        img_np = np.array(img_pil).astype(np.float32)
        blurred_channels = []

        if applied_blur_type == 'motion':
            length = random.uniform(*MOTION_BLUR_LENGTH_RANGE)
            angle = random.uniform(*MOTION_BLUR_ANGLE_RANGE)
            kernel = create_motion_blur_kernel(length, angle)
        else: # defocus
            radius = random.uniform(*DEFOCUS_BLUR_RADIUS_RANGE)
            kernel = create_defocus_blur_kernel(radius)

        # Apply convolution channel by channel
        for i in range(3): # R, G, B
            channel = img_np[:, :, i]
            blurred_channel = scipy.signal.convolve2d(channel, kernel, mode='same', boundary='symm')
            blurred_channels.append(blurred_channel)

        # Stack channels back, clip, and convert to uint8
        blurred_np = np.stack(blurred_channels, axis=-1)
        blurred_np = np.clip(blurred_np, 0, 255)
        blurred_img_pil = Image.fromarray(blurred_np.astype(np.uint8)) # Convert back to PIL

    # - Apply subsequent degradations to the blurred image -
    img_degraded = blurred_img_pil # Start with the blurred image

    # 2. Add Noise
    noise_intensity = random.uniform(*NOISE_INTENSITY_RANGE)
    if noise_intensity > 0:
        img_np = np.array(img_degraded).astype(np.float32)
        noise = np.random.normal(0, noise_intensity, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img_degraded = Image.fromarray(img_np)

    # 3. Simulate JPEG Compression
    jpeg_quality = random.randint(*JPEG_QUALITY_RANGE)
    buffer = io.BytesIO()
    img_degraded.save(buffer, "JPEG", quality=jpeg_quality)
    buffer.seek(0)
    img_degraded = Image.open(buffer).convert("RGB") # Ensure RGB

    return img_degraded


def generate_sample(i, num_total_samples, font_list, corpus_lines):
    """Generates one pair of (Sharp, Blurred) text images."""
    font_path = "N/A" # Initialize for better error reporting
    text = "N/A"
    font_size_px = 0
    font_size_pt = 0
    try:
        # - 1. Select Random Parameters -
        if not font_list:
            return False
        font_path = random.choice(font_list)
        font_size_pt = random.randint(*FONT_SIZE_RANGE_PT)

        if not corpus_lines:
            return False

        # - Select initial lines -
        num_lines_to_select = random.randint(*LINES_PER_SAMPLE)
        num_lines_to_select = min(num_lines_to_select, len(corpus_lines))
        max_start_index = len(corpus_lines) - num_lines_to_select
        if max_start_index < 0: max_start_index = 0
        start_index = random.randint(0, max_start_index)
        selected_raw_lines = corpus_lines[start_index : start_index + num_lines_to_select]

        # - Process selected lines to limit words -
        processed_lines = []
        for line in selected_raw_lines:
            stripped_line = line.strip() # Remove leading/trailing whitespace
            if not stripped_line:
                processed_lines.append("")
                continue

            words = stripped_line.split() # Split into words
            if len(words) > MAX_WORDS_PER_LINE:
                # Truncate if too many words
                truncated_line = " ".join(words[:MAX_WORDS_PER_LINE])
                processed_lines.append(truncated_line)
            else:
                # Keep the line as is if within the word limit
                processed_lines.append(stripped_line)

        # - Combine processed lines into the final text string -
        text = "\n".join(processed_lines).strip() # Join with newline, strip potential leading/trailing newlines

        # - Fallback if processing resulted in empty text -
        if not text:
             text = "Default Text" # Use a fallback

        # Check TOTAL text length
        if len(text) > MAX_RENDER_TEXT_LENGTH:
            print(f"Warning: Skipping sample {i} due to excessive TOTAL text length ({len(text)} > {MAX_RENDER_TEXT_LENGTH}) even after word limiting. Font: {os.path.basename(font_path)}", file=sys.stderr)
            return False

        text_color = tuple(
            random.randint(c_min, c_max) for c_min, c_max in zip(*TEXT_COLOR_RANGE)
        )

        # Determine background type
        r = random.random()
        cum = 0
        chosen_bg_type = 'color'
        for t, p in BACKGROUND_TYPE_PROB.items():
            cum += p
            if r < cum:
                chosen_bg_type = t
                break

        # - 2. Create Background -
        hr_background = create_background(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            chosen_bg_type,
            BG_COLOR_RANGE,
            BG_NOISE_INTENSITY_RANGE,
            TEXTURE_DIR,
        )
        sharp_image = hr_background.copy()
        draw = ImageDraw.Draw(sharp_image)


        # - 3. Load Font -
        font_size_px = int(font_size_pt * 96 / 72)

        if font_size_px <= 0:
             print(f"Warning: Skipping sample {i} due to calculated pixel size <= 0 ({font_size_px}px from {font_size_pt}pt). Font: {os.path.basename(font_path)}", file=sys.stderr)
             return False

        try:
             font = ImageFont.truetype(font_path, font_size_px)
        except IOError as e:
             print(f"Warning: Skipping sample {i} due to font IO error: {e}. Font: {font_path}", file=sys.stderr)
             return False
        except OSError as e:
             print(f"Warning: Skipping sample {i} due to font loading OSError (possibly invalid size): {e}. Font: {font_path}, Size: {font_size_px}px ({font_size_pt}pt)", file=sys.stderr)
             return False


        # - 4. Calculate Text Position -
        text_width = 0
        text_height = 0
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except OSError as e:
            if "execution context too long" in str(e).lower():
                 print(f"Warning: Skipping sample {i} due to text layout timeout. Font: {os.path.basename(font_path)}, Size: {font_size_px}px, Text Length: {len(text)}", file=sys.stderr)
                 return False
            else:
                 print(f"Warning: Skipping sample {i} due to unexpected OSError during textbbox: {e}. Font: {font_path}, Size: {font_size_px}px", file=sys.stderr)
                 return False
        except Exception as e:
             print(f"Warning: Skipping sample {i} due to generic error during textbbox: {e}. Font: {font_path}, Size: {font_size_px}px", file=sys.stderr)
             return False

        if text_width <= 0 or text_height <= 0:
            if len(text) > 0 :
                print(f"Warning: Skipping sample {i} due to invalid text dimensions ({text_width}x{text_height}) after bbox. Font: {os.path.basename(font_path)}, Size: {font_size_px}px", file=sys.stderr)
            return False

        if text_width > IMAGE_WIDTH or text_height > IMAGE_HEIGHT:
             pass

        max_x = IMAGE_WIDTH - text_width
        max_y = IMAGE_HEIGHT - text_height
        text_x_offset = random.randint(0, max(0, max_x))
        text_y_offset = random.randint(0, max(0, max_y))
        draw_x = text_x_offset - text_bbox[0]
        draw_y = text_y_offset - text_bbox[1]


        # - 5. Render Sharp Text -
        try:
            draw.text((draw_x, draw_y), text, font=font, fill=text_color)
        except Exception as e:
            print(f"Warning: Skipping sample {i} due to error during final text drawing: {e}. Font: {os.path.basename(font_path)}, Size: {font_size_px}px", file=sys.stderr)
            return False


        # - 6. Apply Degradations for BLURRED image -
        blurred_image = apply_degradation_for_deblur(sharp_image)


        # - 7. Save Images -
        width = len(str(num_total_samples))
        base_filename = f"sample_{i:0{width}d}"
        sharp_image.save(os.path.join(OUTPUT_SHARP_DIR, f"{base_filename}_sharp.png"), "PNG", compress_level=9)
        blurred_image.save(os.path.join(OUTPUT_BLURRED_DIR, f"{base_filename}_blurred.png"), "PNG", compress_level=9)

        return True

    # - Exception Handling -
    except Exception as e:
        print(f"\n--- Error generating sample {i} ---", file=sys.stderr)
        print(f"Exception type: {type(e).__name__}", file=sys.stderr)
        print(f"Error message: {e}", file=sys.stderr)
        print(f"Font path: {font_path}", file=sys.stderr)
        print(f"Font size: {font_size_px}px ({font_size_pt}pt)", file=sys.stderr)
        print(f"Text snippet (first 100 chars): {text[:100]}...", file=sys.stderr)
        print(f"Text length: {len(text) if text != 'N/A' else 'N/A'}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("---------------------------------\n", file=sys.stderr)
        return False

# - Main Execution Block -
if __name__ == "__main__":
    print(f"Script started at: {datetime.datetime.now()}")
    print(f"Outputting Sharp images to: {os.path.abspath(OUTPUT_SHARP_DIR)}")
    print(f"Outputting Blurred images to: {os.path.abspath(OUTPUT_BLURRED_DIR)}")

    # Create output directories
    os.makedirs(OUTPUT_SHARP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BLURRED_DIR, exist_ok=True)

    # - Pre-load resources to pass to workers -
    font_list = []
    try:
        font_list = [
             os.path.join(FONT_DIR, f)
             for f in os.listdir(FONT_DIR)
             if f.lower().endswith((".ttf", ".otf"))
         ]
        if not font_list:
            raise FileNotFoundError(f"No font files found in {FONT_DIR}")
        print(f"Found {len(font_list)} fonts.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)
    except Exception as e:
        print(f"Error listing fonts: {e}", file=sys.stderr)
        exit(1)

    corpus_lines = []
    try:
        with open(TEXT_CORPUS, "r", encoding="utf-8") as f:
            corpus_lines = f.readlines()
        if not corpus_lines:
            raise ValueError("Text corpus file is empty.")
        print(f"Loaded {len(corpus_lines)} lines from corpus.")
    except FileNotFoundError:
        print(f"Error: Text corpus file '{TEXT_CORPUS}' not found.", file=sys.stderr)
        exit(1)
    except ValueError as e:
         print(f"Error: {e}", file=sys.stderr)
         exit(1)
    except Exception as e:
        print(f"Error reading corpus: {e}", file=sys.stderr)
        exit(1)

    # - Generate Samples using Multiprocessing -
    print(f"\nGenerating {NUM_SAMPLES} sample..")
    success_count = 0
    failed_count = 0

    # Sequential Version with tqdm
    for i in tqdm(range(NUM_SAMPLES), desc="Generating Samples", unit="sample"):
        while True:
            try:
                result = generate_sample(i, NUM_SAMPLES, font_list, corpus_lines)
                if result:
                    success_count += 1
                    break
                else:
                    failed_count += 1
            except OSError as e:
                print(f"Warning: OSError {e} occurred. Retrying...", file=sys.stderr)
                continue

    # - Final Summary -
    print("\n-----------------------------------------")
    print(f"Sample generation finished at: {datetime.datetime.now()}")
    print(f"Successfully generated: {success_count} / {NUM_SAMPLES}")
    if failed_count > 0:
        print(f"Failed or skipped:    {failed_count} / {NUM_SAMPLES}")
    print(f"Sharp images saved in: {os.path.abspath(OUTPUT_SHARP_DIR)}")
    print(f"Blurred images saved in: {os.path.abspath(OUTPUT_BLURRED_DIR)}")
    print("-----------------------------------------")