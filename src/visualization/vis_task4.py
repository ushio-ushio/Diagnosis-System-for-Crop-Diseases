import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_all_reports(report_dir):
    reports = []
    report_files = Path(report_dir).glob("sample_*_report.json")
    for file in report_files:
        with open(file, 'r', encoding='utf-8') as f:
            reports.append(json.load(f))
    return reports


def plot_task_correlation(reports, output_path):
    type_confs = [r['type_confidence'] for r in reports]
    severity_confs = [r['severity_confidence'] for r in reports]
    severities = [r['severity'] for r in reports]

    severity_colors = {'Healthy': 0, 'General Disease': 1, 'Serious Disease': 2}
    colors = [severity_colors[s] for s in severities]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(type_confs, severity_confs, c=colors, cmap='viridis', alpha=0.7, s=100)
    plt.xlabel('Disease Type Confidence')
    plt.ylabel('Severity Confidence')
    plt.title('Correlation between Disease Type and Severity Confidence')
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Severity Level')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(reports, output_path):
    type_confs = [r['type_confidence'] for r in reports]
    severity_confs = [r['severity_confidence'] for r in reports]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(type_confs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Disease Type Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Disease Type Confidence')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(severity_confs, bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Severity Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Severity Confidence')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_disease_severity_heatmap(reports, output_path):
    diseases = list(set(r['disease'] for r in reports))
    severities = ['Healthy', 'General Disease', 'Serious Disease']

    count_matrix = np.zeros((len(severities), len(diseases)))
    for r in reports:
        disease_idx = diseases.index(r['disease'])
        severity_idx = severities.index(r['severity'])
        count_matrix[severity_idx, disease_idx] += 1

    plt.figure(figsize=(15, 6))
    im = plt.imshow(count_matrix, cmap='YlOrRd', aspect='auto')
    plt.xticks(range(len(diseases)), diseases, rotation=45, ha='right')
    plt.yticks(range(len(severities)), severities)
    plt.xlabel('Disease Types')
    plt.ylabel('Severity Levels')
    plt.title('Distribution of Disease Types across Severity Levels')

    for i in range(len(severities)):
        for j in range(len(diseases)):
            plt.text(j, i, int(count_matrix[i, j]),
                    ha='center', va='center', color='black')

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    inference_results_dir = project_root / "outputs" / "task4" / "inference_results"
    visualization_dir = project_root / "outputs" / "task4" / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    reports = load_all_reports(inference_results_dir)

    if not reports:
        print("No reports found. Please run inference first.")
        return

    plot_task_correlation(reports, visualization_dir / "task_correlation.png")
    print("Generated task correlation plot")

    plot_confidence_distribution(reports, visualization_dir / "confidence_distribution.png")
    print("Generated confidence distribution plot")

    plot_disease_severity_heatmap(reports, visualization_dir / "disease_severity_heatmap.png")
    print("Generated disease-severity heatmap")

    print(f"All visualizations saved to {visualization_dir}")


if __name__ == "__main__":
    main()
