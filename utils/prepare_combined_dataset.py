import os
import shutil
from PIL import Image
from tqdm import tqdm

# Configuration
SOURCE_PATHS = {
    'train': ['/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/train_A',
              '/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/train_B'],
    'val': ['/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/hold_A',
            '/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/hold_B'],
    'test': ['/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/test_A',
             '/home/ensta/data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/test_B']
}

TARGET_DIR = '/home/ensta/ensta-sidi/Project_IA/combined_dataset'
MAX_IMAGES = {
    'train': 2000,
    'val': 400,
    'test': 400
}  # Adjusted to keep storage reasonable
DOWNSAMPLE_FACTOR = 4

os.makedirs(TARGET_DIR, exist_ok=True)

for split, paths in SOURCE_PATHS.items():
    hr_dir = os.path.join(TARGET_DIR, f'{split}_HR')
    lr_dir = os.path.join(TARGET_DIR, f'{split}_LR')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    images_a = sorted(os.listdir(paths[0]))
    images_b = sorted(os.listdir(paths[1]))

    selected_images_a = images_a[:MAX_IMAGES[split] // 2]
    selected_images_b = images_b[:MAX_IMAGES[split] // 2]

    combined_images = selected_images_a + selected_images_b

    for img_name in tqdm(combined_images, desc=f'Processing {split}'):
        source_dir = paths[0] if img_name in selected_images_a else paths[1]
        hr_img_path = os.path.join(source_dir, img_name)

        # Copy HR image
        target_hr_path = os.path.join(hr_dir, img_name)
        shutil.copy(hr_img_path, target_hr_path)

        # Create and save LR image
        img = Image.open(hr_img_path)
        w, h = img.size
        lr_img = img.resize((w // DOWNSAMPLE_FACTOR, h // DOWNSAMPLE_FACTOR), Image.BICUBIC)
        lr_img.save(os.path.join(lr_dir, img_name))

print("Dataset preparation completed.")
