import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.datasets.agri_disease import AgriDiseaseDataset
from src.datasets.transforms import get_val_transform
from src.utils.gradcam import GradCAM


def build_model(num_classes=61):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def plot_gradcam(origin, cam, save_path):
    fig, ax = plt.subplots(2, 1, figsize=(5, 8))
    ax[0].imshow(origin)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(origin)
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    save_dir = project_root / "outputs" / "vis_task1"
    save_dir.mkdir(parents=True, exist_ok=True)

    val_list = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"
    val_dataset = AgriDiseaseDataset(val_list, transform=get_val_transform(224))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = 61
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes)
    best_model_path = project_root / "outputs" / "task1_resnet50_best.pth"
    obj = torch.load(best_model_path, map_location=device)

    if isinstance(obj, dict) and "model_state" in obj:
        model.load_state_dict(obj["model_state"])
    else:
        model.load_state_dict(obj)

    model.to(device)
    model.eval()
    print(f"[Load] Loaded best model: {best_model_path}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_dir / "task1_confusion_matrix.png")
    plt.close()
    print("[Save] Confusion matrix saved.")

    N = 20
    indices = random.sample(range(len(val_dataset)), N)

    target_layer = model.layer4[-1]
    cam_gen = GradCAM(model, target_layer)

    for i, idx in enumerate(indices):
        img_tensor, label = val_dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        H, W = img_tensor.shape[1], img_tensor.shape[2]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        img_input.requires_grad_(True)
        scores = model(img_input)
        pred_class = scores.argmax(1).item()

        cam = cam_gen(scores, pred_class, (H, W))

        save_path = save_dir / f"gradcam_{i}_label{label}_pred{pred_class}.png"
        plot_gradcam(img_np, cam, save_path)
        print(f"[Save] Grad-CAM -> {save_path}")


if __name__ == "__main__":
    main()
