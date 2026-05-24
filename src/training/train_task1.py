import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

from src.datasets.agri_disease import AgriDiseaseDataset
from src.datasets.transforms import get_train_transform, get_val_transform
from src.models.task1_resnet import build_model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = (scaler is not None) and (device.type == "cuda")

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"

    train_list = data_dir / "AgriculturalDisease_trainingset" / "train_list.txt"
    val_list = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"

    batch_size = 32
    num_workers = 4
    num_classes = 61
    num_epochs = 50
    lr = 1e-3
    resume = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_dataset = AgriDiseaseDataset(train_list, transform=get_train_transform(224))
    val_dataset = AgriDiseaseDataset(val_list, transform=get_val_transform(224))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == "cuda" else None

    save_dir = project_root / "outputs"
    save_dir.mkdir(exist_ok=True)

    best_model_path = save_dir / "task1_resnet50_best.pth"
    checkpoint_path = save_dir / "task1_resnet50_checkpoint.pth"
    log_path = save_dir / "task1_train_log.csv"

    best_val_acc = 0.0
    start_epoch = 1

    if resume:
        loaded = False
        if checkpoint_path.is_file():
            print(f"[Resume] Found checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                if "scaler_state" in ckpt and ckpt["scaler_state"] is not None and scaler is not None:
                    scaler.load_state_dict(ckpt["scaler_state"])
                best_val_acc = ckpt.get("best_val_acc", 0.0)
                last_epoch = ckpt.get("epoch", 0)
                start_epoch = last_epoch + 1 if last_epoch > 0 else 1
                loaded = True
            else:
                model.load_state_dict(ckpt)
                start_epoch = 1
                loaded = True

        if (not loaded) and best_model_path.is_file():
            obj = torch.load(best_model_path, map_location=device)
            if isinstance(obj, dict) and "model_state" in obj:
                model.load_state_dict(obj["model_state"])
                best_val_acc = obj.get("best_val_acc", 0.0)
            else:
                model.load_state_dict(obj)
            loaded = True

        if not loaded:
            print("[Resume] WARNING: resume=True but no checkpoint found, training from scratch.")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:03d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        write_header = not log_path.exists()
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                           f"{val_loss:.6f}", f"{val_acc:.6f}", f"{best_val_acc:.6f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "epoch": epoch,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
            }, checkpoint_path)
            print(f"[Checkpoint] Saved best model (acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
