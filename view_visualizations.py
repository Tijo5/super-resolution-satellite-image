import os
import glob
from PIL import Image

output_dir = "/home/ensta/ensta-sidi/Project_IA/visualization_output_20/"
image_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))

for image_path in image_paths:
    img = Image.open(image_path)
    img.show()
    input(f"Affichage de {os.path.basename(image_path)} — appuie sur Entrée pour voir la suivante...")
