import argparse
import cv2
import matplotlib.pyplot as plt
import os
from preprocessing import preprocessing

import albumentations as A

def visualize_augmentations(image_path, transform, n=5, output_dir=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, n+1, figsize=(4*(n+1), 4))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    for i in range(n):
        augmented = transform(image=image)['image']
        axes[i+1].imshow(augmented.permute(1,2,0).cpu().numpy())
        axes[i+1].set_title(f'Augmented {i+1}')
        axes[i+1].axis('off')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f'augmented_{i+1}.png')
            cv2.imwrite(out_path, cv2.cvtColor(augmented.permute(1,2,0).cpu().numpy()*255, cv2.COLOR_RGB2BGR))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image augmentations.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--n', type=int, default=5, help='Number of augmentations to show')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save augmented images')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'], help='Use train or val transform')
    args = parser.parse_args()

    transform = preprocessing.train_transform if args.mode == 'train' else preprocessing.val_transform
    visualize_augmentations(args.image_path, transform, n=args.n, output_dir=args.output_dir)
