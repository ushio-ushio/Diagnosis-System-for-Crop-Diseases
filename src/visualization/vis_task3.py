import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from torch import amp
import torch.nn.functional as F

from src.datasets.transforms import get_val_transform
from src.datasets.severity_mapping import SEVERITY3_TO_ID, get_severity3_id
from src.utils.gradcam import GradCAM


class Severity3VisDataset(Dataset):
    def __init__(self, list_file: Path, transform=None):
        self.samples = []
        self.transform = transform

        list_file = Path(list_file)
        data_root = list_file.parent.parent

        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path_str, label_str = line.split()
                path_str = path_str.replace("\\", "/")
                img_path = data_root / path_str
                disease_class = int(label_str)
                severity_id = get_severity3_id(disease_class)
                self.samples.append((img_path, severity_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sev = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sev, str(img_path)


def build_task3_model(ckpt_path: Path, device: torch.device):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print(f"[Model] Loaded Task3 checkpoint from {ckpt_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total params: {total_params / 1e6:.2f}M")
    return model


@torch.no_grad()
def evaluate_and_plot_cm(model, loader, device, save_dir: Path, use_amp: bool = True):
    model.eval()
    num_classes = 3

    all_labels = []
    all_preds = []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp)):
            outputs = model(imgs)
        _, preds = outputs.max(1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    all_labels = np.array(all_labels, dtype=np.int64)
    all_preds = np.array(all_preds, dtype=np.int64)

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
    acc = float((all_labels == all_preds).mean())

    id2name_en = {0: "Healthy", 1: "Mild", 2: "Severe"}

    print("\n=== Task3 Validation Metrics ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    for cid in range(num_classes):
        cname = id2name_en[cid]
        print(f"Class {cid} ({cname}): Recall={per_class_recall[cid]:.4f}, F1={per_class_f1[cid]:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=[id2name_en[i] for i in range(num_classes)],
        yticklabels=[id2name_en[i] for i in range(num_classes)],
        ylabel="True label",
        xlabel="Predicted label",
        title="Task3 Severity Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    save_path = save_dir / "task3_confusion_matrix.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Viz] Confusion matrix saved to: {save_path}")


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.4):
    H, W = cam.shape
    img = img_pil.resize((W, H))
    img_np = np.array(img).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]

    overlay = heatmap * alpha + img_np * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    return overlay


def make_compare_image(orig_img: Image.Image, overlay_np: np.ndarray):
    H, W, _ = overlay_np.shape
    orig_resized = orig_img.resize((W, H))
    orig_np = np.array(orig_resized, dtype=np.uint8)

    canvas = np.zeros((2 * H, W, 3), dtype=np.uint8)
    canvas[0:H, :, :] = orig_np
    canvas[H:2 * H, :, :] = overlay_np
    return canvas


def generate_gradcam_examples(model, dataset, device, save_dir: Path, samples_per_class: int = 3):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    id2name_en = {0: "Healthy", 1: "Mild", 2: "Severe"}
    num_classes = len(id2name_en)

    counters = {cid: 0 for cid in range(num_classes)}

    target_layer = model.layer4[-1].conv3
    cam_extractor = GradCAM(model, target_layer)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if all(counters[cid] >= samples_per_class for cid in counters):
            break

        img_tensor, sev, img_path = dataset[idx]
        sev_id = int(sev)
        if counters[sev_id] >= samples_per_class:
            continue

        orig_img = Image.open(img_path).convert("RGB")
        input_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)

        with amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(input_tensor)
        _, pred = outputs.max(1)
        pred_id = int(pred.item())

        cam, _ = cam_extractor.generate_cam(input_tensor, task='type', target_class=pred_id)
        overlay = overlay_cam_on_image(orig_img, cam, alpha=0.4)
        compare_np = make_compare_image(orig_img, overlay)

        cname_true = id2name_en[sev_id]
        cname_pred = id2name_en[pred_id]
        base_name = Path(img_path).stem

        save_subdir = save_dir / f"class_{sev_id}_{cname_true}"
        os.makedirs(save_subdir, exist_ok=True)
        save_path = save_subdir / f"{base_name}_pred{pred_id}_{cname_pred}.png"

        Image.fromarray(compare_np).save(save_path)
        counters[sev_id] += 1
        print(f"[GradCAM] Saved: {save_path}")

    print("[GradCAM] Done. Samples per class:", counters)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    vis_dir = outputs_dir / "vis_task3"
    vis_dir.mkdir(exist_ok=True)

    val_list = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"
    ckpt_path = outputs_dir / "task3_severity3_resnet50_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    val_transform = get_val_transform(224)
    val_dataset = Severity3VisDataset(val_list, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=64,
        shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_task3_model(ckpt_path, device)

    evaluate_and_plot_cm(model, val_loader, device, vis_dir, use_amp=True)

    gradcam_dir = vis_dir / "gradcam"
    generate_gradcam_examples(model, val_dataset, device, gradcam_dir, samples_per_class=10)


if __name__ == "__main__":
    main()
