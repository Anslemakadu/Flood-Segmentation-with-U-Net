import random
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as T

class SyntheticFloodDataset(Dataset):
    """
    Synthetic satellite-like RGB images + binary flood masks.
    - length: number of samples the dataset reports
    - size: square tile size (H=W=size)
    - img_transform / mask_transform: optional torchvision transforms
    - pre_generate: if True, pre-generate samples (useful for deterministic val)
    - seed: optional seed for reproducibility
    """
    def __init__(self, length=100, size=128,
                 img_transform=None, mask_transform=None,
                 pre_generate=False, seed=None):
        self.length = length
        self.size = size

        # resize -> to tensor -> ImageNet normalize
        if img_transform is None:
            self.img_transform = T.Compose([
                T.Resize((self.size, self.size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
            ])
        else:
            self.img_transform = img_transform

        # default mask transform (resize -> to tensor) - no normalization
        if mask_transform is None:
            self.mask_transform = T.Compose([
                T.Resize((self.size, self.size)),
                T.ToTensor()   # yields [1,H,W] floats in [0,1]
            ])
        else:
            self.mask_transform = mask_transform

        # reproducibility
        self.pre_generate = bool(pre_generate)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # pre-generate all samples 
        if self.pre_generate:
            self._samples = [self._make_sample() for _ in range(self.length)]
        else:
            self._samples = None

    def __len__(self):
        return self.length

    def _make_sample(self):
        s = self.size
        # base: random terrain color (brown/greenish)
        base = Image.new('RGB', (s, s),
                         (random.randint(30,160),
                          random.randint(100,200),
                          random.randint(30,160)))
        draw = ImageDraw.Draw(base)

        # mask: single-channel 'L' mode, background 0
        mask = Image.new('L', (s, s), 0)
        mdraw = ImageDraw.Draw(mask)

        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            x0 = random.randint(0, s//2)
            y0 = random.randint(0, s//2)
            x1 = random.randint(s//2, s)
            y1 = random.randint(s//2, s)

            if random.random() < 0.4:
                # river-like curved line
                steps = random.randint(3,6)
                pts = []
                for i in range(steps):
                    px = int(x0 + (x1-x0) * i/(steps-1) + random.randint(-8,8))
                    py = int(y0 + (y1-y0) * i/(steps-1) + random.randint(-8,8))
                    px = max(0, min(s-1, px))
                    py = max(0, min(s-1, py))
                    pts.append((px, py))
                width = random.randint(8,16)
                mdraw.line(pts, fill=255, width=width)
                draw.line(pts, fill=(30,90,160), width=width)
            else:
                # pond / lake
                x0, y0 = max(0,x0), max(0,y0)
                x1, y1 = min(s-1,x1), min(s-1,y1)
                mdraw.ellipse([x0,y0,x1,y1], fill=255)
                draw.ellipse([x0,y0,x1,y1], fill=(30,90,160))

        # add small gaussian noise to the RGB image
        np_img = np.array(base).astype(np.int16)
        noise = (np.random.randn(*np_img.shape) * 6).astype(np.int16)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        base = Image.fromarray(np_img)

        return base, mask

    def __getitem__(self, idx):
        # get PIL pair 
        if self._samples is not None:
            img_pil, mask_pil = self._samples[idx]
        else:
            img_pil, mask_pil = self._make_sample()

        # transforms - image normalized, mask converted to tensor only
        img = self.img_transform(img_pil)      # [3,H,W], normalized floats
        mask = self.mask_transform(mask_pil)   # [1,H,W], floats in [0,1]
        mask = (mask > 0.5).float()            # ensure strict binary 0/1

        return img, mask
