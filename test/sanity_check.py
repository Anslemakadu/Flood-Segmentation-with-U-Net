# sanity_check.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SyntheticFloodDataset
from model import UNetSmall
from utils import BCEDiceLoss, iou_score

def run_sanity(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    print("Device:", device)
    
    ds = SyntheticFloodDataset(length=16, size=128, pre_generate=True, seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = UNetSmall(in_channels=3, out_channels=1, base_c=32).to(device)
    criterion = BCEDiceLoss()

    batch = next(iter(loader))
    imgs, masks = batch
    imgs, masks = imgs.to(device), masks.to(device)

    print("img shape:", imgs.shape)   # expect [B,3,128,128]
    print("mask shape:", masks.shape) # expect [B,1,128,128]

    # forward
    logits = model(imgs)
    print("logits shape:", logits.shape)  # expect [B,1,128,128]

    # loss
    loss = criterion(logits, masks)
    print("loss:", loss.item())

    # metric
    iou = iou_score(logits, masks)
    print("IoU (before step):", iou)

    # backward (test gradients)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    # grad exists on a parameter
    grad_exists = any(p.grad is not None for p in model.parameters())
    print("grad exists after backward:", grad_exists)
    optimizer.step()

if __name__ == "__main__":
    run_sanity()
