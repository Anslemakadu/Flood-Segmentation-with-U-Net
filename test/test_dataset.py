import matplotlib.pyplot as plt
import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import SyntheticFloodDataset


def unnormalize(img_tensor):
    # img_tensor: [3,H,W] normalized with ImageNet mean/std
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    img = img_tensor.permute(1,2,0).cpu().numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img

if __name__ == '__main__':
    ds = SyntheticFloodDataset(length=8, size=128, pre_generate=True, seed=42)
    for i in range(8):
        img, mask = ds[i]
        img_np = unnormalize(img)
        mask_np = mask.squeeze(0).cpu().numpy()

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow(img_np); plt.title('Image'); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(mask_np, cmap='gray'); plt.title('Mask'); plt.axis('off')
        plt.show()
