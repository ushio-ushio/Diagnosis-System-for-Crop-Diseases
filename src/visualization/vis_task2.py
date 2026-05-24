import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from thop import profile
from pathlib import Path

from src.models.se_resnet import SEBlock, SEBasicBlock
from src.models.task2_se_resnet import build_inference_model


class AgriculturalDiseaseDataset(Dataset):
    def __init__(self, img_root, list_file, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.samples = []

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                full_rel_path = parts[0]
                label = int(parts[1])
                img_name = os.path.basename(full_rel_path)
                img_path = os.path.join(self.img_root, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                else:
                    print(f"Warning: Image not found, skipped: {img_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 61
    MODEL_PATH = project_root / "best_model_fewshot.pth"

    VAL_IMG_DIR = data_dir / "AgriculturalDisease_validationset" / "images"
    VAL_LIST_FILE = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading validation set...")
    val_dataset = AgriculturalDiseaseDataset(
        img_root=str(VAL_IMG_DIR),
        list_file=str(VAL_LIST_FILE),
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"  Validation samples: {len(val_dataset)}")

    print("Building model (layer3=5 blocks, reduction=32)...")
    model = build_inference_model(num_classes=NUM_CLASSES)

    state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

    print("\nComputing model complexity...")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"  Params: {params / 1e6:.2f} M")
    print(f"  FLOPs: {flops / 1e9:.2f} G")
    if params > 20e6:
        print("  WARNING: Params exceed 20M limit!")
    else:
        print("  Params符合 <=20M requirement.")

    print("\nGenerating confusion matrix...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix (Total Samples: {len(all_labels)})')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

    try:
        from torchcam.methods import GradCAM as TorchCAM
        from torchcam.utils import overlay_mask
        from torchvision.transforms.functional import to_pil_image

        print("\nGenerating Grad-CAM visualization...")
        cam_extractor = TorchCAM(model, target_layer="layer4")

        demo_indices = [20, 40, 68, 90, 100]
        for i, idx in enumerate(demo_indices):
            if idx >= len(val_dataset):
                continue
            img_path, true_label = val_dataset.samples[idx]
            original_img = Image.open(img_path).convert('RGB')

            input_tensor = val_transform(original_img).unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            pred_label = output.argmax().item()

            activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
            heatmap = to_pil_image(activation_map[0].squeeze(0), mode='F')
            result = overlay_mask(original_img.resize((224, 224)), heatmap, alpha=0.6)

            plt.figure(figsize=(6, 10))
            plt.subplot(2, 1, 1)
            plt.imshow(original_img)
            plt.title(f"Original\nTrue: {true_label}, Pred: {pred_label}")
            plt.axis('off')

            plt.subplot(2, 1, 2)
            plt.imshow(result)
            plt.title("Grad-CAM")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'combined_{i}.png', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved combined_{i}.png")
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        print("  Suggest: pip install torchcam")

    print("\nAll visualizations completed!")
