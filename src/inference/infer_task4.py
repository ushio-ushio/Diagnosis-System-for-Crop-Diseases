import os
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt

from src.models.task4_resnet import MultiTaskResNet
from src.datasets.task4_dataset import MultiTaskDataset
from src.datasets.transforms import get_val_transform
from src.utils.gradcam import GradCAM
from src.inference.report_generator import DiagnosisReport


def visualize_results(original_img, type_heatmap, severity_heatmap, report, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(original_img)
    axes[1].imshow(type_heatmap, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Disease Type Focus: {report['disease']}")
    axes[1].axis('off')

    axes[2].imshow(original_img)
    axes[2].imshow(severity_heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Severity Focus: {report['severity']}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / "outputs" / "task4" / "best_model.pth"
    data_dir = project_root / "data"
    test_list = data_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"
    output_dir = project_root / "outputs" / "task4" / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskResNet(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = get_val_transform(224)

    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    dummy_dataset = MultiTaskDataset(project_root / "data" / "AgriculturalDisease_trainingset" / "train_list.txt")
    report_generator = DiagnosisReport(dummy_dataset)

    test_dataset = MultiTaskDataset(test_list, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    type_count = {}
    severity_count = {}
    MIN_SAMPLES_PER_TYPE = 50
    MAX_SAMPLES_PER_TYPE = 80
    MAX_TOTAL_SAMPLES = 1000

    all_types = set([item[1] for item in test_dataset.samples])
    print(f"[INFO] Detected {len(all_types)} disease classes")

    for i, (img_tensor, type_label, severity_label) in enumerate(test_loader):
        type_label_val = type_label.item() if hasattr(type_label, 'item') else type_label
        severity_label_val = severity_label.item() if hasattr(severity_label, 'item') else severity_label

        if type_label_val not in type_count:
            type_count[type_label_val] = 0
        if severity_label_val not in severity_count:
            severity_count[severity_label_val] = 0

        if type_count[type_label_val] >= MAX_SAMPLES_PER_TYPE:
            if all(c >= MIN_SAMPLES_PER_TYPE for c in type_count.values()):
                print(f"[INFO] All {len(all_types)} classes reached minimum sample count ({MIN_SAMPLES_PER_TYPE}), stopping")
                break
            continue

        type_count[type_label_val] += 1

        completed_count = sum(1 for c in type_count.values() if c >= MIN_SAMPLES_PER_TYPE)
        if i % 50 == 0 or completed_count == len(all_types):
            print(f"[PROGRESS] Sample {i}: {completed_count}/{len(all_types)} classes达标")

        if all(c >= MIN_SAMPLES_PER_TYPE for c in type_count.values()):
            print(f"[INFO] All {len(all_types)} classes reached minimum sample count, stopping")
            break

        if i >= MAX_TOTAL_SAMPLES:
            print(f"[WARNING] Reached max sample limit ({MAX_TOTAL_SAMPLES}), stopping early")
            break

        img_path = test_dataset.samples[i][0]
        img_tensor = img_tensor.to(device)
        type_label = type_label.item() if hasattr(type_label, 'item') else type_label
        severity_label = severity_label.item() if hasattr(severity_label, 'item') else severity_label

        with torch.no_grad():
            type_pred, severity_pred = model(img_tensor)
            type_prob = torch.softmax(type_pred, dim=1)[0]
            severity_prob = torch.softmax(severity_pred, dim=1)[0]
            type_conf, type_class = torch.max(type_prob, dim=0)
            severity_conf, severity_class = torch.max(severity_prob, dim=0)

        type_heatmap, _ = gradcam.generate_cam(img_tensor, task='type', target_class=type_class)
        severity_heatmap, _ = gradcam.generate_cam(img_tensor, task='severity', target_class=severity_class)

        original_img = Image.open(img_path).convert("RGB")

        import cv2
        type_heatmap_resized = cv2.resize(type_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        severity_heatmap_resized = cv2.resize(severity_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)

        report = report_generator.generate_report(
            img_path, type_class.item(), severity_class.item(),
            type_conf.item(), severity_conf.item(), None, None
        )

        type_heatmap_path = output_dir / f"sample_{i}_type_heatmap.jpg"
        severity_heatmap_path = output_dir / f"sample_{i}_severity_heatmap.jpg"
        visualization_path = output_dir / f"sample_{i}_visualization.jpg"

        plt.imsave(str(type_heatmap_path), type_heatmap_resized, cmap='jet')
        plt.imsave(str(severity_heatmap_path), severity_heatmap_resized, cmap='jet')
        plt.close('all')

        try:
            visualize_results(original_img, type_heatmap_resized, severity_heatmap_resized, report, visualization_path)
        except Exception as e:
            print(f"Failed to save visualization {visualization_path.name}: {e}")

        report["type_heatmap"] = str(type_heatmap_path)
        report["severity_heatmap"] = str(severity_heatmap_path)
        report["visualization"] = str(visualization_path)

        disease_name = report['disease'].replace(' ', '_').replace('/', '_')
        sample_idx = type_count[type_label_val]
        safe_disease_name = re.sub(r'[^\w\-_\. ]', '', disease_name)
        report_filename = f"{safe_disease_name}_sample_{sample_idx}_report.json"

        try:
            with open(output_dir / report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save report {report_filename}: {e}")

        results.append(report)

        if all(c >= MAX_SAMPLES_PER_TYPE for c in type_count.values()):
            break

    try:
        with open(output_dir / "all_reports.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved all reports: all_reports.json")
    except Exception as e:
        print(f"Failed to save all reports: {e}")

    print(f"[INFO] Processing completed, results saved to {output_dir}")


if __name__ == "__main__":
    main()
