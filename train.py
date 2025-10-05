import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import SyntheticFloodDataset
from model import UNetSmall
from utils import BCEDiceLoss, iou_score


def train_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    transform_img = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # datasets
    train_ds = SyntheticFloodDataset(length=200, size=128, img_transform=transform_img, pre_generate=False, seed=42)
    val_ds   = SyntheticFloodDataset(length=50, size=128, img_transform=transform_img, pre_generate=True, seed=123)
    test_ds  = SyntheticFloodDataset(length=30, size=128, img_transform=transform_img, pre_generate=True, seed=456)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = UNetSmall(in_channels=3, out_channels=1, base_c=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss()

    best_iou = 0.0
    epochs = 10
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_iou += iou_score(logits, masks)
        val_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - val IoU: {val_iou:.4f}")

        # save best
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_unet.pth')
            print('Saved best model, IoU:', best_iou)

    print('Training done. Best IoU:', best_iou)



    # Test Evaluation
    print("\nRunning on test set...")
    model.load_state_dict(torch.load("best_unet.pth", map_location=device))
    model.eval()
    total_iou = 0.0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            total_iou += iou_score(logits, masks)

    test_iou = total_iou / len(test_loader)
    print("Final Test IoU:", test_iou)


if __name__ == '__main__':
    train_loop()
