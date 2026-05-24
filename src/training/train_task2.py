import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from src.models.se_resnet import SEBlock, SEBasicBlock
from src.datasets.agri_disease import AgriDiseaseDataset

SEED = 42
NUM_CLASSES = 61
MAX_K_PER_CLASS = 10
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING_EPS = 0.1
MIXUP_ALPHA = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def create_fewshot_subset(dataset, max_k_per_class=10):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    selected_indices = []
    class_sample_counts = {}

    for cls in range(NUM_CLASSES):
        if cls in class_indices:
            indices = class_indices[cls]
            k = min(max_k_per_class, len(indices))
            selected = random.sample(indices, k)
            selected_indices.extend(selected)
            class_sample_counts[cls] = k
        else:
            class_sample_counts[cls] = 0

    low_classes = [cls for cls, cnt in class_sample_counts.items() if 0 < cnt < max_k_per_class]
    missing_classes = [cls for cls, cnt in class_sample_counts.items() if cnt == 0]

    print(f"Few-shot training set built:")
    print(f"  Total samples: {len(selected_indices)}")
    print(f"  Covered classes: {len([c for c in class_sample_counts.values() if c > 0])}/{NUM_CLASSES}")
    if low_classes:
        print(f"  Classes with <{max_k_per_class} samples: {low_classes[:10]}{'...' if len(low_classes)>10 else ''}")
    if missing_classes:
        print(f"  Missing classes: {missing_classes[:10]}{'...' if len(missing_classes)>10 else ''}")

    return Subset(dataset, selected_indices), class_sample_counts


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"

    train_img_dir = data_dir / "AgriculturalDisease_trainingset" / "images"
    train_list_file = data_dir / "AgriculturalDisease_trainingset" / "train_list.txt"
    val_img_dir = data_dir / "AgriculturalDisease_validationset" / "images"
    val_list_file = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading training set...")
    full_train_dataset = AgriDiseaseDataset(train_list_file, transform=train_transform)
    print(f"  Total training samples: {len(full_train_dataset)}")

    print("Loading validation set...")
    val_dataset = AgriDiseaseDataset(val_list_file, transform=val_transform)
    print(f"  Total validation samples: {len(val_dataset)}")

    fewshot_train_dataset, class_counts = create_fewshot_subset(full_train_dataset, max_k_per_class=MAX_K_PER_CLASS)

    train_loader = DataLoader(fewshot_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "resnet18-f37072fd.pth")

    model = models.resnet18(weights=None)

    model.layer3 = nn.Sequential(
        SEBasicBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ), reduction=32),
        *[SEBasicBlock(256, 256, reduction=32) for _ in range(4)]
    )

    model.layer4 = nn.Sequential(
        SEBasicBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        ), reduction=32),
        *[SEBasicBlock(512, 512, reduction=32) for _ in range(2)]
    )

    model.fc = nn.Linear(512, NUM_CLASSES)

    if os.path.exists(WEIGHTS_PATH):
        try:
            pretrained_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape and 'fc' not in k and 'se' not in k
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded local ResNet18 pretrained weights (SE layers random init): {WEIGHTS_PATH}")
        except Exception as e:
            print(f"Failed to load weights, using random init: {e}")
    else:
        print(f"Weights file not found: {WEIGHTS_PATH}")

    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False
    for layer in [model.layer3, model.layer4, model.fc]:
        for param in layer.parameters():
            param.requires_grad = True

    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING_EPS)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    patience = 10
    trigger_times = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=range(NUM_CLASSES))

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model_fewshot.pth")
            print(f"New best model saved with Val Acc: {best_acc:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"No improvement for {trigger_times} epochs.")

        if trigger_times >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\nTraining finished. Best Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    from pathlib import Path
    main()
