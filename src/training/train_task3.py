import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
import numpy as np

from src.datasets.agri_disease import AgriDiseaseDataset
from src.datasets.transforms import get_train_transform, get_val_transform
from src.datasets.severity_mapping import get_severity3_id, SEVERITY3_TO_ID


class Severity3Dataset(AgriDiseaseDataset):
    def __getitem__(self, idx):
        img, disease_class = super().__getitem__(idx)
        severity_id = get_severity3_id(disease_class)
        return img, severity_id


def build_model_from_task1(num_severity_classes, task1_ckpt_path, device):
    base = models.resnet50(pretrained=False)
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, 61)

    ckpt = torch.load(task1_ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    base.load_state_dict(state, strict=False)
    print(f"[Model] Loaded Task1 checkpoint from {task1_ckpt_path}")

    base.fc = nn.Linear(in_features, num_severity_classes)
    total_params = sum(p.numel() for p in base.parameters())
    print(f"[Model] Total params: {total_params / 1e6:.2f}M")
    return base


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
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
def evaluate(model, loader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / total
    acc = correct / total

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    per_class_recall = []
    per_class_precision = []
    per_class_f1 = []

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class_recall.append(recall)
        per_class_precision.append(precision)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))
    return avg_loss, acc, macro_f1, per_class_recall, per_class_f1, cm


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"

    train_list = data_dir / "AgriculturalDisease_trainingset" / "train_list.txt"
    val_list = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"
    task1_ckpt_path = project_root / "outputs" / "task1_resnet50_best.pth"

    batch_size = 64
    num_workers = 4
    num_severity_classes = 3
    num_epochs = 40
    base_lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_dataset = Severity3Dataset(train_list, transform=get_train_transform(224))
    val_dataset = Severity3Dataset(val_list, transform=get_val_transform(224))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model_from_task1(
        num_severity_classes=num_severity_classes,
        task1_ckpt_path=task1_ckpt_path,
        device=device,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    save_dir = project_root / "outputs"
    save_dir.mkdir(exist_ok=True)
    best_macro_f1 = 0.0

    severity_id2name = {v: k for k, v in SEVERITY3_TO_ID.items()}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, macro_f1, per_class_recall, per_class_f1, cm = evaluate(
            model, val_loader, criterion, device, num_classes=num_severity_classes
        )

        print(f"[Epoch {epoch:03d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} MacroF1: {macro_f1:.4f}")

        for cid in range(num_severity_classes):
            cname = severity_id2name[cid]
            rec = per_class_recall[cid]
            f1 = per_class_f1[cid]
            print(f"    Class {cid} ({cname}): Recall={rec:.4f}, F1={f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            ckpt_path = save_dir / "task3_severity3_resnet50_best.pth"
            torch.save({
                "model_state": model.state_dict(),
                "best_macro_f1": best_macro_f1,
                "best_val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"[Checkpoint] Saved best Task3 model (MacroF1={best_macro_f1:.4f})")

    print(f"[Done] Best Macro-F1 on validation: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
