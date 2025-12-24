import os, torch, sys
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

sys.path.append('..')
from seed import seed
from utils.io import load_config, save_image
from utils.dct import apply_dct_batch, apply_idct_batch
from utils.masks import get_mask

def frequency_swap(config):
    seed(config['seed'])

    # Create swapped directory if not exist
    os.makedirs(config['data']['swapped_dir'], exist_ok=True)

    real_dir = config['data']['real_dir']
    recon_dir = config['data']['recon_dir']

    image_files = [
        f for f in os.listdir(real_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Frequency Swapping"):
            # Load real and reconstructed images

            real_img = Image.open(os.path.join(real_dir, img_file)).convert('RGB')
            recon_img = Image.open(os.path.join(recon_dir, img_file)).convert('RGB')

            # tensor 
            real_tensor = transform(real_img).unsqueeze(0)
            recon_tensor = transform(recon_img).unsqueeze(0)

            # Apply DCT:
            real_dct = apply_dct_batch(real_tensor)
            recon_dct = apply_dct_batch(recon_tensor)

            # Create masks 
            _, _, h, w = real_dct.shape
            lf_mask = get_mask(
                config['masks']['type'],
                h, w,
                config['masks']['cutoff_ratio']
            )
            hf_mask = 1 - lf_mask

            # freq swap : real HF + Recon LF
            swapped_dct = real_dct * hf_mask + recon_dct * lf_mask

            # Inverse DCT
            swapped_img = apply_idct_batch(swapped_dct)
            swapped_img = torch.clamp(swapped_img, 0, 1)

            # Save Swapped Image
            save_image(
                       swapped_img, 
                       os.path.join(config['data']['swapped_dir'], 
                                   f"swapped_{img_file}")
                    )

if __name__ == "__main__":
    config = load_config("../config.yaml")
    frequency_swap(config)