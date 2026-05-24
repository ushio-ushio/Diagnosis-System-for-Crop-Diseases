import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import torchvision.utils

from src.datasets.task4_dataset import MultiTaskDataset
from src.models.task4_resnet import MultiTaskResNet
from src.datasets.transforms import get_train_transform, get_val_transform
from src.utils.gradcam import GradCAM
from src.inference.report_generator import DiagnosisReport


def train_one_epoch(model, loader, criterion_type, criterion_severity,
                   optimizer, device, epoch, num_epochs, alpha=1.0, beta=1.2):
    model.train()
    total_loss = 0.0
    type_correct = 0
    severity_correct = 0
    total_samples = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader),
                       desc=f'Epoch {epoch+1}/{num_epochs}',
                       unit='batch', ncols=100)
    for batch_idx, (imgs, type_labels, severity_labels) in progress_bar:
        imgs = imgs.to(device)
        type_labels = type_labels.to(device) if hasattr(type_labels, 'to') else type_labels
        severity_labels = severity_labels.to(device) if hasattr(severity_labels, 'to') else severity_labels

        optimizer.zero_grad()

        type_pred, severity_pred = model(imgs)
        loss_type = criterion_type(type_pred, type_labels)
        loss_severity = criterion_severity(severity_pred, severity_labels)
        total_loss_batch = alpha * loss_type + beta * loss_severity

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item() * imgs.size(0)
        _, type_preds = torch.max(type_pred, 1)
        _, severity_preds = torch.max(severity_pred, 1)

        type_correct += (type_preds == type_labels).sum().item()
        severity_correct += (severity_preds == severity_labels).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    type_acc = type_correct / total_samples
    severity_acc = severity_correct / total_samples
    return avg_loss, type_acc, severity_acc


@torch.no_grad()
def evaluate(model, loader, criterion_type, criterion_severity,
            device, num_type_classes=61, num_severity_classes=3):
    model.eval()
    total_loss = 0.0
    type_correct = 0
    severity_correct = 0
    total_samples = 0

    type_conf_matrix = np.zeros((num_type_classes, num_type_classes), dtype=np.int64)
    severity_conf_matrix = np.zeros((num_severity_classes, num_severity_classes), dtype=np.int64)

    val_progress = tqdm(enumerate(loader), total=len(loader),
                      desc='Validating', unit='batch', ncols=100, leave=False)
    for batch_idx, (imgs, type_labels, severity_labels) in val_progress:
        imgs = imgs.to(device)
        type_labels = type_labels.to(device) if hasattr(type_labels, 'to') else type_labels
        severity_labels = severity_labels.to(device) if hasattr(severity_labels, 'to') else severity_labels

        type_pred, severity_pred = model(imgs)
        loss_type = criterion_type(type_pred, type_labels)
        loss_severity = criterion_severity(severity_pred, severity_labels)
        total_loss_batch = loss_type + 1.2 * loss_severity

        total_loss += total_loss_batch.item() * imgs.size(0)

        _, type_preds = torch.max(type_pred, 1)
        _, severity_preds = torch.max(severity_pred, 1)

        for t, p in zip(type_labels.cpu().numpy(), type_preds.cpu().numpy()):
            type_conf_matrix[t, p] += 1
        for t, p in zip(severity_labels.cpu().numpy(), severity_preds.cpu().numpy()):
            severity_conf_matrix[t, p] += 1

        type_correct += (type_preds == type_labels).sum().item()
        severity_correct += (severity_preds == severity_labels).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    type_acc = type_correct / total_samples
    severity_acc = severity_correct / total_samples
    severity_macro_f1 = calculate_macro_f1(severity_conf_matrix)

    return avg_loss, type_acc, severity_acc, severity_macro_f1, type_conf_matrix, severity_conf_matrix


def calculate_macro_f1(conf_matrix):
    num_classes = conf_matrix.shape[0]
    f1_scores = []
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "task4"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list = data_dir / "AgriculturalDisease_trainingset" / "train_list.txt"
    val_list = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"

    train_dataset = MultiTaskDataset(train_list, transform=get_train_transform(224))
    val_dataset = MultiTaskDataset(val_list, transform=get_val_transform(224))

    batch_size = 32
    num_workers = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet(pretrained=True).to(device)

    criterion_type = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 30
    best_macro_f1 = 0.0

    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    report_generator = DiagnosisReport(val_dataset)

    print("[INFO] Starting multi-task model training...")

    for epoch in range(1, num_epochs + 1):
        train_loss, type_acc, severity_acc = train_one_epoch(
            model, train_loader, criterion_type, criterion_severity,
            optimizer, device, epoch=epoch-1, num_epochs=num_epochs
        )

        val_loss, val_type_acc, val_severity_acc, macro_f1, _, _ = evaluate(
            model, val_loader, criterion_type, criterion_severity, device
        )

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Type Acc: {type_acc:.4f}, Severity Acc: {severity_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Type Acc: {val_type_acc:.4f}, Severity Acc: {val_severity_acc:.4f}, Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_macro_f1': best_macro_f1,
                'type_accuracy': val_type_acc,
                'severity_accuracy': val_severity_acc
            }, output_dir / "best_model.pth", _use_new_zipfile_serialization=False)
            print(f"[INFO] New best model saved, Macro-F1: {best_macro_f1:.4f}")

            generate_sample_reports(model, val_loader, gradcam, report_generator,
                                  device, output_dir, num_samples=5)

    print(f"[INFO] Training completed. Best validation Macro-F1: {best_macro_f1:.4f}")


def generate_sample_reports(model, data_loader, gradcam, report_generator,
                          device, output_dir, num_samples=5):
    model.eval()
    sample_dir = output_dir / "sample_reports"
    sample_dir.mkdir(exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    count = 0
    reports = []
    type_count = {}
    severity_count = {}
    max_samples_per_type = 10

    for imgs, type_labels, severity_labels in data_loader:
        if count >= num_samples and all(c >= max_samples_per_type for c in type_count.values()):
            break

        imgs = imgs.to(device)
        type_labels = type_labels.to(device)
        severity_labels = severity_labels.to(device)

        for i in range(min(num_samples - count, imgs.size(0))):
            img = imgs[i:i+1]
            type_label = type_labels[i].item()
            severity_label = severity_labels[i].item()

            type_count[type_label] = type_count.get(type_label, 0)
            severity_count[severity_label] = severity_count.get(severity_label, 0)

            if type_count[type_label] >= max_samples_per_type:
                continue

            type_count[type_label] += 1

            with torch.no_grad():
                type_pred, severity_pred = model(img)
                type_prob = torch.softmax(type_pred, dim=1)[0]
                severity_prob = torch.softmax(severity_pred, dim=1)[0]
                type_conf, type_class = torch.max(type_prob, dim=0)
                severity_conf, severity_class = torch.max(severity_prob, dim=0)

            type_heatmap, _ = gradcam.generate_cam(img, task='type', target_class=type_class)
            severity_heatmap, _ = gradcam.generate_cam(img, task='severity', target_class=severity_class)

            type_heatmap_path = sample_dir / f"sample_{count}_type_heatmap.jpg"
            severity_heatmap_path = sample_dir / f"sample_{count}_severity_heatmap.jpg"

            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.imshow(type_heatmap, cmap='jet')
            plt.colorbar()
            plt.title(f"Type Heatmap - Class: {type_class.item()}")
            plt.savefig(str(type_heatmap_path), bbox_inches='tight', dpi=100)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.imshow(severity_heatmap, cmap='jet')
            plt.colorbar()
            plt.title(f"Severity Heatmap - Class: {severity_class.item()}")
            plt.savefig(str(severity_heatmap_path), bbox_inches='tight', dpi=100)
            plt.close()

            original_img_path = sample_dir / f"sample_{count}_original.jpg"
            img_denorm = img.cpu() * std[:, None, None] + mean[:, None, None]
            img_denorm = torch.clamp(img_denorm, 0, 1)
            torchvision.utils.save_image(img_denorm, original_img_path)

            report = report_generator.generate_report(
                str(original_img_path),
                type_class.item(),
                severity_class.item(),
                type_conf.item(),
                severity_conf.item(),
                str(type_heatmap_path),
                str(severity_heatmap_path)
            )
            reports.append(report)
            count += 1

    with open(sample_dir / "sample_reports.json", 'w') as f:
        json.dump(reports, f, indent=2)

    print(f"[INFO] Generated {len(reports)} sample diagnostic reports in {sample_dir}")


if __name__ == "__main__":
    main()
