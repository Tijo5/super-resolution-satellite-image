import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

inference_results_dir = "/home/ensta/ensta-sidi/Project_IA/inference_results/"
output_dir = "/home/ensta/ensta-sidi/Project_IA/satlas-super-resolution-main/visualization_output_20/"

os.makedirs(output_dir, exist_ok=True)

# Trouver tous les dossiers d'inférence
inference_folders = sorted(glob.glob(os.path.join(inference_results_dir, '*')))
print(f"Nombre de dossiers d'inférence trouvés : {len(inference_folders)}")

# On ne prend que les 20 premiers
inference_folders = inference_folders[:20]

for folder in inference_folders:
    lr_path = os.path.join(folder, "lr.png")
    sr_path = os.path.join(folder, "sr.png")

    lr_img = Image.open(lr_path)
    sr_img = Image.open(sr_path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title("Low Resolution")
    axs[0].axis("off")

    axs[1].imshow(sr_img)
    axs[1].set_title("Super Resolved")
    axs[1].axis("off")

    chip_name = os.path.basename(folder)
    save_path = os.path.join(output_dir, f"visualization_{chip_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

print(f"Visualisations pour 20 images enregistrées dans : {output_dir}")
