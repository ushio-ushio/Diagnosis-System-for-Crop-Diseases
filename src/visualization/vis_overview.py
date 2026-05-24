import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter


def load_all_reports(report_dir):
    reports = []
    report_files = list(Path(report_dir).glob("*_sample_*_report.json"))
    print(f"Found {len(report_files)} report files")

    for file in report_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reports.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load report file {file}: {e}")

    print(f"Successfully loaded {len(reports)} reports")
    return reports


def plot_disease_distribution(reports, output_path):
    disease_counter = Counter([r['disease'] for r in reports])
    disease_items = {k: v for k, v in disease_counter.items() if "Healthy" not in k}

    if not disease_items:
        print("Warning: Not enough disease data to generate distribution chart")
        return

    plt.figure(figsize=(12, 8))
    patches, texts, autotexts = plt.pie(disease_items.values(), labels=disease_items.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Crop Disease Distribution', fontsize=16, pad=20)

    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_severity_distribution(reports, output_path):
    severity_counter = Counter([r['severity'] for r in reports])

    severities = ['Healthy', 'General Disease', 'Serious Disease']
    counts = [severity_counter.get(s, 0) for s in severities]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(severities, counts, color=['green', 'orange', 'red'], alpha=0.7)
    plt.xlabel('Disease Severity Level', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Disease Severity Distribution', fontsize=16, pad=20)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_boxplot(reports, output_path):
    disease_types = list(set([r['disease'] for r in reports]))
    disease_types = [d for d in disease_types if "Healthy" not in d]

    if not disease_types:
        print("Warning: Not enough disease data to generate confidence boxplot")
        return

    type_conf_data = []
    severity_conf_data = []
    labels = []

    for disease in disease_types:
        disease_reports = [r for r in reports if r['disease'] == disease]
        if len(disease_reports) > 1:
            type_confidences = [r['type_confidence'] for r in disease_reports]
            severity_confidences = [r['severity_confidence'] for r in disease_reports]
            type_conf_data.append(type_confidences)
            severity_conf_data.append(severity_confidences)
            labels.append(disease)

    if not labels:
        print("Warning: Not enough data to generate confidence boxplot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    bp1 = ax1.boxplot(type_conf_data, labels=labels, patch_artist=True)
    ax1.set_title('Confidence Distribution for Disease Types', fontsize=14)
    ax1.set_ylabel('Confidence', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    bp2 = ax2.boxplot(severity_conf_data, labels=labels, patch_artist=True)
    ax2.set_title('Confidence Distribution for Severity Levels', fontsize=14)
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_treatment_advice_distribution(reports, output_path):
    advice_counter = Counter([r['treatment_advice'] for r in reports])

    if not advice_counter:
        print("Warning: No treatment advice data to generate distribution chart")
        return

    plt.figure(figsize=(10, 8))
    patches, texts, autotexts = plt.pie(advice_counter.values(), labels=advice_counter.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Treatment Recommendations', fontsize=16, pad=20)

    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_summary(reports, output_path):
    if len(reports) == 0:
        print("Warning: No report data to generate performance summary chart")
        return

    avg_type_conf = np.mean([r['type_confidence'] for r in reports])
    avg_severity_conf = np.mean([r['severity_confidence'] for r in reports])

    severity_counter = Counter([r['severity'] for r in reports])
    total_samples = len(reports)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    conf_types = ['Disease Type Recognition', 'Severity Assessment']
    avg_confs = [avg_type_conf, avg_severity_conf]
    bars1 = ax1.bar(conf_types, avg_confs, color=['skyblue', 'salmon'])
    ax1.set_ylim(0, 1)
    ax1.set_title('Average Confidence Comparison', fontsize=14)
    ax1.set_ylabel('Average Confidence')

    for bar, conf in zip(bars1, avg_confs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')

    severities = ['Healthy', 'General Disease', 'Serious Disease']
    counts = [severity_counter.get(s, 0) for s in severities]
    percentages = [count/total_samples*100 for count in counts]
    bars2 = ax2.bar(severities, percentages, color=['green', 'orange', 'red'])
    ax2.set_title('Percentage by Severity Level', fontsize=14)
    ax2.set_ylabel('Percentage (%)')

    for bar, pct in zip(bars2, percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    type_confs = [r['type_confidence'] for r in reports]
    severity_confs = [r['severity_confidence'] for r in reports]
    ax3.hist(type_confs, bins=20, alpha=0.7, label='Disease Type', color='blue', edgecolor='black')
    ax3.hist(severity_confs, bins=20, alpha=0.7, label='Severity', color='red', edgecolor='black')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Distribution Histogram')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    metrics = ['Type Recognition', 'Severity Assessment', 'Average Confidence', 'Healthy Detection', 'Serious Disease Detection']
    values = [
        avg_type_conf,
        avg_severity_conf,
        (avg_type_conf + avg_severity_conf) / 2,
        severity_counter.get('Healthy', 0) / total_samples,
        severity_counter.get('Serious Disease', 0) / total_samples
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax4.fill(angles, values, alpha=0.25, color='blue')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_title('Model Performance Radar Chart', fontsize=14, pad=20)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    inference_results_dir = project_root / "outputs" / "task4" / "inference_results"
    visualization_dir = project_root / "outputs" / "task4" / "enhanced_visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    report_files = list(Path(inference_results_dir).glob("*_sample_*_report.json"))
    if not report_files:
        print(f"Error: No report files found. Check directory: {inference_results_dir}")
        return

    reports = load_all_reports(inference_results_dir)

    if not reports:
        print("Error: Failed to load any report files")
        return

    print(f"Loaded {len(reports)} reports")

    plot_disease_distribution(reports, visualization_dir / "disease_distribution.png")
    print("Generated disease distribution chart")

    plot_severity_distribution(reports, visualization_dir / "severity_distribution.png")
    print("Generated severity distribution chart")

    plot_confidence_boxplot(reports, visualization_dir / "confidence_boxplot.png")
    print("Generated confidence boxplot")

    plot_treatment_advice_distribution(reports, visualization_dir / "treatment_advice_distribution.png")
    print("Generated treatment advice distribution chart")

    plot_performance_summary(reports, visualization_dir / "performance_summary.png")
    print("Generated performance summary chart")

    print(f"All enhanced visualizations saved to {visualization_dir}")


if __name__ == "__main__":
    main()
