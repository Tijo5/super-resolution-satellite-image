import os
import random
from PIL import Image, UnidentifiedImageError

# Paths
source_base = "/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/naip_xbd/images"
dest_base = "/home/ensta/ensta-sidi/Project_IA/combined_dataset"

num_crops_per_image = 5  # Number of random crops per HR image
crop_size_hr = 128
scale = 4
crop_size_lr = crop_size_hr // scale

num_images_per_split = {
    "train": 2000,
    "val": 1000,
    "test": 1000
}

splits = {
    "train": ["train_A", ""],
    "val": ["hold_A", ""],
    "test": ["test_A", ""]
}

def random_crop(img, crop_size):
    w, h = img.size
    if w < crop_size or h < crop_size:
        raise ValueError(f"Image too small for crop: {w}x{h}")
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    return img.crop((left, top, left + crop_size, top + crop_size))

def downsample_image(hr_img):
    return hr_img.resize((crop_size_lr, crop_size_lr), Image.BICUBIC)

for split, (dir_a, dir_b) in splits.items():
    hr_dir = os.path.join(dest_base, split, "HR")
    lr_dir = os.path.join(dest_base, split, "LR")

    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    all_files = []
    for sub_dir in [dir_a, dir_b]:
        src = os.path.join(source_base, sub_dir)
        files = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.png')]
        all_files.extend(files)

    selected_files = random.sample(all_files, min(num_images_per_split[split], len(all_files)))

    for idx, src_file in enumerate(selected_files):
        try:
            img = Image.open(src_file)
            img.load()  # Force loading to catch truncated files
            for crop_idx in range(num_crops_per_image):
                try:
                    crop = random_crop(img, crop_size_hr)
                    hr_filename = f"{idx}_{crop_idx}_HR.png"
                    hr_filepath = os.path.join(hr_dir, hr_filename)
                    crop.save(hr_filepath)

                    lr_img = downsample_image(crop)
                    lr_filename = f"{idx}_{crop_idx}_LR.png"
                    lr_filepath = os.path.join(lr_dir, lr_filename)
                    lr_img.save(lr_filepath)
                except Exception as e:
                    print(f"Skipping crop from image {src_file}: {e}")
        except (OSError, UnidentifiedImageError) as e:
            print(f"Skipping corrupted image {src_file}: {e}")

print("Combined cropped dataset creation and downsampling completed.")
