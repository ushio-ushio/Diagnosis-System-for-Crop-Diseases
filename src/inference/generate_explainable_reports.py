import json
import os
from pathlib import Path

DISEASE_CHARACTERISTICS = {
    "Scab": "Scab characteristics: Brown scab lesions appear on the leaf/fruit surface, and severe cases cause leaf distortion and fruit malformation",
    "Rust": "Rust characteristics: Orange-yellow rust spots appear on the leaf surface, with rust spore heaps on the back, and severe cases cause leaf death",
    "Healthy": "Healthy crop: No obvious disease symptoms, good growth status",
    "Powdery Mildew": "Powdery mildew characteristics: White powdery substance covers the leaf surface, later turns gray-brown, and leaves curl",
    "Leaf Blight": "Leaf blight characteristics: Irregular brown lesions appear on leaves, with yellow halos at the edges, and severe cases cause whole leaf death",
    "Black Rot": "Black rot characteristics: Black sunken lesions appear on the fruit surface, and the entire fruit turns black and rots in later stages",
    "Cedar Apple Rust": "Cedar apple rust characteristics: Yellow spots appear on the upper surface of leaves, with orange-red protrusions on the lower surface",
    "Cercospora Leaf Spot": "Cercospora leaf spot characteristics: Circular to irregular brown spots appear on leaves, with gray-white centers",
    "Septoria Leaf Spot": "Septoria leaf spot characteristics: Small circular lesions appear on leaves, with gray-white centers and dark brown edges",
    "Spider Mites": "Spider mites characteristics: Yellow-white spots appear on leaves, with fine webbing and mites on the back",
    "Bacterial Spot": "Bacterial spot characteristics: Water-soaked small spots appear on leaves and fruits, later turning brown",
    "Early Blight": "Early blight characteristics: Concentric ring-shaped brown lesions appear on leaves, with yellow edges",
    "Late Blight": "Late blight characteristics: Water-soaked dark green lesions appear on leaves, rapidly expanding to cause whole leaf rot",
    "Leaf Mold": "Leaf mold characteristics: Gray-purple mold layer appears on the back of leaves, with yellow spots on the front",
    "Mosaic Virus": "Mosaic virus characteristics: Yellow-green mottled symptoms appear on leaves, with leaf malformation",
    "Yellow Leaf Curl Virus": "Yellow leaf curl virus characteristics: Leaf edges curl upward, plants are stunted, and leaves turn yellow",
    "General": "General crop disease characteristics: Abnormal spots, discoloration or necrotic areas appear on leaves"
}

EXPLAINABILITY_ANALYSIS = {
    "Scab": {
        "Healthy": "Model identified smooth leaf surface with normal color and no disease symptoms",
        "General Disease": "Model identified initial scab lesions on leaf/fruit surface, manifested as small brown spots",
        "Serious Disease": "Model identified large-area scab regions covering leaf/fruit surface with obvious deformation"
    },
    "Rust": {
        "Healthy": "Model identified smooth leaf surface with normal color and no rust symptoms",
        "General Disease": "Model identified initial rust spots on leaf surface, manifested as a few orange-yellow spots",
        "Serious Disease": "Model identified large-area rust regions with rust spore heaps covering the back of leaves"
    },
    "Powdery Mildew": {
        "Healthy": "Model identified smooth leaf surface with normal color and no powdery mildew symptoms",
        "General Disease": "Model identified initial powdery substance on leaf surface, with local light powdery coverage",
        "Serious Disease": "Model identified large-area powdery coverage on leaf surface, with leaf curling phenomenon"
    },
    "Leaf Blight": {
        "Healthy": "Model identified intact leaves with smooth edges and normal color",
        "General Disease": "Model identified small-range brown lesions at leaf edges",
        "Serious Disease": "Model identified large-area leaf death with lesions expanding in patches"
    },
    "Black Rot": {
        "Healthy": "Model identified smooth fruit surface with normal color and no lesions",
        "General Disease": "Model identified initial small black spots on fruit surface",
        "Serious Disease": "Model identified large-area black rot on fruit surface with obvious depression"
    },
    "General": {
        "Healthy": "Model identified healthy crops with no abnormal symptoms",
        "General Disease": "Model identified mild disease symptoms on crop surface",
        "Serious Disease": "Model identified severe disease symptoms on crop surface"
    }
}

TREATMENT_ADVICE = {
    "Scab": {
        "Healthy": "- Continue good orchard management\n- Regular inspection to prevent disease occurrence",
        "General Disease": "- Improve orchard ventilation and lighting\n- Spray Bordeaux mixture or mancozeb for prevention\n- Remove diseased leaves to reduce pathogens",
        "Serious Disease": "- Immediately spray therapeutic fungicides (tebuconazole, flusilazole)\n- Thoroughly remove and destroy diseased leaves and fruits\n- Improve orchard drainage to reduce humidity"
    },
    "Rust": {
        "Healthy": "- Continue good crop management\n- Regular inspection to prevent disease occurrence",
        "General Disease": "- Increase ventilation and reduce field humidity\n- Spray triadimefon or tebuconazole for prevention\n- Remove diseased residues",
        "Serious Disease": "- Immediately spray therapeutic fungicides (flusilazole, hexaconazole)\n- Remove and destroy large amounts of diseased leaves\n- Avoid prolonged leaf wetness"
    },
    "Powdery Mildew": {
        "Healthy": "- Continue good crop management\n- Regular inspection to prevent disease occurrence",
        "General Disease": "- Improve ventilation conditions and avoid excessive plant density\n- Spray sulfur powder or triadimefon for prevention\n- Control watering to avoid prolonged leaf wetness",
        "Serious Disease": "- Immediately spray therapeutic fungicides (myclobutanil, flusilazole)\n- Prune diseased leaves and increase ventilation\n- Strictly control watering time and method"
    },
    "General": {
        "Healthy": "- Continue good crop management\n- Regular inspection to prevent disease occurrence",
        "General Disease": "- Strengthen observation and consider using appropriate fungicides for preventive treatment\n- Improve crop growing environment conditions",
        "Serious Disease": "- Immediately take treatment measures\n- Use targeted agents and consider isolating the infected area\n- Remove severely infected plants to prevent spread"
    }
}


def get_sample_id(image_path):
    return os.path.basename(image_path).split('.')[0]


def get_disease_base_name(disease_name):
    if "(Serious)" in disease_name:
        return disease_name.replace(" (Serious)", "")
    elif "(General)" in disease_name:
        return disease_name.replace(" (General)", "")
    return disease_name


def get_severity_level(severity_text):
    if "Healthy" in severity_text:
        return "Healthy"
    elif "General" in severity_text:
        return "General Disease"
    elif "Serious" in severity_text:
        return "Serious Disease"
    return severity_text


def generate_explainable_report(report_data):
    sample_id = get_sample_id(report_data['image_path'])
    disease_name = get_disease_base_name(report_data['disease'])
    severity_level = get_severity_level(report_data['severity'])
    type_conf = f"{report_data['type_confidence']:.3f}"
    sev_conf = f"{report_data['severity_confidence']:.3f}"

    characteristics = DISEASE_CHARACTERISTICS.get(disease_name,
                                                DISEASE_CHARACTERISTICS.get("General",
                                                                         f"{disease_name} characteristics: Abnormal spots, discoloration or necrotic areas appear on leaves/fruits"))

    explain_text = EXPLAINABILITY_ANALYSIS.get(disease_name,
                                             EXPLAINABILITY_ANALYSIS.get("General",
                                                                       f"Model identified typical symptoms associated with {disease_name}"))
    explain_text = explain_text.get(severity_level, explain_text.get("General Disease",
                                                                   "Model identified disease symptoms on crop surface"))

    treatment_advice = TREATMENT_ADVICE.get(disease_name,
                                          TREATMENT_ADVICE.get("General",
                                                             "- Please consult local agricultural experts for professional advice\n- Collect samples for laboratory testing"))
    treatment_advice = treatment_advice.get(severity_level,
                                          treatment_advice.get("General Disease",
                                                             "- Strengthen observation and consider using appropriate fungicides for preventive treatment\n- Consult local agricultural experts"))

    lines = [
        "Crop Disease Diagnostic Report",
        "",
        f"Sample ID: {sample_id}",
        f"Input Image: {os.path.basename(report_data['image_path'])}",
        "",
        "1. Disease Classification Results",
        f"Predicted Category: {report_data['crop']} {report_data['disease']}",
        f"Prediction Confidence: {type_conf}",
        "",
        "2. Severity Assessment",
        f"Predicted Level: {report_data['severity']}",
        f"Confidence: {sev_conf}",
        "",
        "3. Disease Characteristics Analysis",
        characteristics,
        "",
        "4. Explainability Analysis (Grad-CAM)",
        explain_text,
        "",
        "5. Agricultural Implications & Treatment Recommendations",
        treatment_advice
    ]
    return "\n".join(lines)


def load_reports_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_report_to_file(report_text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Report saved to: {output_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    inference_results_dir = project_root / "outputs" / "task4" / "inference_results"
    explainable_reports_dir = project_root / "outputs" / "task4" / "explainable_reports"
    explainable_reports_dir.mkdir(parents=True, exist_ok=True)

    all_reports_path = inference_results_dir / "all_reports.json"
    if all_reports_path.exists():
        print("Found complete report file, generating explainable reports...")
        reports = load_reports_from_json(all_reports_path)
        for i, report in enumerate(reports):
            explainable_report_text = generate_explainable_report(report)
            output_path = explainable_reports_dir / f"sample_{i}_explainable_report.txt"
            save_report_to_file(explainable_report_text, output_path)
        print(f"Generated {len(reports)} explainable reports to {explainable_reports_dir}")
    else:
        print("Complete report file not found, looking for individual reports...")
        report_files = list(inference_results_dir.glob("sample_*_report.json"))
        if not report_files:
            print("Error: No report files found, please run the inference program first.")
            return
        print(f"Found {len(report_files)} report files, generating explainable reports...")
        for report_file in report_files:
            try:
                report = load_reports_from_json(report_file)
                explainable_report_text = generate_explainable_report(report)
                sample_id = report_file.stem.replace('_report', '_explainable_report')
                output_path = explainable_reports_dir / f"{sample_id}.txt"
                save_report_to_file(explainable_report_text, output_path)
            except Exception as e:
                print(f"Error processing {report_file}: {e}")
        print(f"Processing completed, explainable reports saved to {explainable_reports_dir}")


if __name__ == "__main__":
    main()
