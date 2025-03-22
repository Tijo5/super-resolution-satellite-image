# Super-Resolution Satellite Image Project 🚀  

## Description  
Ce projet implémente une pipeline complète de super-résolution d’images satellites en utilisant des réseaux antagonistes génératifs (GAN), inspiré du projet [Satlas Super-Resolution](https://github.com/allenai/satlas-super-resolution).  
Nous avons entraîné notre modèle **(from scratch)** sur le dataset [EIE - Earth Intelligence Engine](https://huggingface.co/datasets/blutjens/eie-earth-intelligence-engine).  

## Dataset utilisé  
- **Hautes résolutions (HR)** : Images provenant du dataset EIE, uniquement les images **pré-disaster** (dossiers `*_A`) afin de garantir des données homogènes.  
- **Basses résolutions (LR)** : Générées à partir des images HR via un script spécifique reproduisant les effets d'images dégradées par inpainting (bruit, flou, correction gamma, perte de saturation, patch noise).  

## Pipeline d'entraînement  
Nous avons suivi les étapes décrites dans le dépôt officiel [Satlas Super-Resolution](https://github.com/allenai/satlas-super-resolution) avec les ajustements suivants :  
- Génération du dataset combiné (HR et LR) à partir des images EIE.  
- Entraînement d’un modèle **SSR-ESRGAN** from scratch.  
- Configurations adaptées à notre nombre d’images et au format LR/HR (512×512 → 1024×1024).  

## Comment reproduire notre travail   

### 1. Installation  
- Cloner le dépôt et installer les dépendances :  
  ```bash
  git clone https://github.com/Tijo5/super-resolution-satellite-image.git
  cd super-resolution-satellite-image
  pip install -r requirements.txt

### 2. Générer le dataset combiné

Adaptez et exécutez le script `ssr/utils/create_combined_dataset.py` pour générer les images LR et HR à partir du dossier **EIE (pré-disaster)**.

---

### 3. Entraîner le modèle

- Lancement de l'entraînement :  

  ```bash
  python -m ssr.train -opt ssr/options/esrgan_s2naip_urban.yml

### 4. Inference
- Une fois le modèle entraîné, lancez l’inférence sur de nouvelles images LR avec :
  ```bash
  python -m ssr.infer -opt ssr/options/infer_example.yml
- Les images super-résolues seront stockées dans le dossier défini dans save_path.

### 5. Références

- [Satlas Super-Resolution - GitHub officiel du projet](https://github.com/allenai/satlas-super-resolution)  
  Ce repository présente le framework complet de super-résolution pour les images satellites, développé par AllenAI. Il inclut le code, les configurations, et les étapes d’entraînement utilisées dans ce projet.

- [Dataset EIE (Earth Intelligence Engine) - Hugging Face](https://huggingface.co/datasets/blutjens/eie-earth-intelligence-engine)  
  Il s’agit du jeu de données haute résolution que nous avons utilisé et recadré pour générer nos images HR. À partir de ce dataset, des images basse résolution (LR) ont été créées via un processus contrôlé de downsampling et de dégradation afin d'entraîner le modèle.
  
