import random
from datetime import datetime

SEVERITY_LEVELS = ["Healthy", "General Disease", "Serious Disease"]

DISEASE_FEATURES = {
    "Scab (General)": "Olive-green to black spots on leaves and fruit, velvety texture on undersides of leaves.",
    "Scab (Serious)": "Extensive black lesions covering most of leaf surface, fruit deformation and premature drop.",
    "Frogeye Spot": "Circular purple spots with gray centers on leaves, can cause significant defoliation.",
    "Cedar Apple Rust (General)": "Yellow-orange spots on upper leaf surface with tube-like structures underneath.",
    "Cedar Apple Rust (Serious)": "Severe yellowing and premature leaf drop, extensive orange fungal growth on both sides of leaves.",
    "Powdery Mildew (General)": "White powdery coating on young leaves and shoots, mild leaf distortion.",
    "Powdery Mildew (Serious)": "Heavy white fungal growth covering leaves, severe leaf curling and stunted growth.",
    "Cercospora Leaf Spot (General)": "Small gray lesions with reddish borders on leaves, minimal yield impact.",
    "Cercospora Leaf Spot (Serious)": "Numerous coalescing lesions causing significant leaf blighting and yield reduction.",
    "Corn Rust (General)": "Small reddish-brown pustules on leaves, mostly on lower leaves.",
    "Corn Rust (Serious)": "Extensive pustules covering all leaf surfaces, rapid plant senescence.",
    "Corn Curvularia Leaf Spot (General)": "Small oval lesions with dark borders, moderate leaf damage.",
    "Corn Curvularia Leaf Spot (Serious)": "Large coalescing lesions causing severe leaf blighting and stalk weakness.",
    "Maize Dwarf Mosaic Virus": "Yellow streaking along veins, stunted growth, mosaic patterns on leaves.",
    "Bacterial Spot (General)": "Small water-soaked spots on leaves and fruit, limited spread.",
    "Bacterial Spot (Serious)": "Large coalescing lesions causing leaf drop and significant fruit blemishes.",
    "Early Blight (General)": "Concentric ring lesions on older leaves, moderate defoliation.",
    "Early Blight (Serious)": "Rapid spread to upper leaves, severe defoliation and yield loss.",
    "Late Blight (General)": "Water-soaked lesions with white fungal growth on undersides, spreading rapidly.",
    "Late Blight (Serious)": "Complete field infection, plant collapse within days under wet conditions.",
    "Leaf Mold (General)": "Yellow spots on upper leaf surface, olive-green mold underneath.",
    "Leaf Mold (Serious)": "Severe yellowing and leaf drop, significant yield reduction.",
    "Septoria Leaf Spot (General)": "Small circular spots with dark borders, mostly on lower leaves.",
    "Septoria Leaf Spot (Serious)": "Extensive spotting causing premature leaf death and defoliation.",
    "Spider Mites (General)": "Fine webbing, light stippling on leaves.",
    "Spider Mites (Serious)": "Heavy webbing, bronzed leaves, severe plant stress.",
    "Yellow Leaf Curl Virus": "Severe leaf curling, yellowing, and stunting of new growth.",
    "Mosaic Virus": "Mosaic patterns of light and dark green on leaves, stunted growth.",
    "Unknown": "Specific features could not be determined. Further examination recommended."
}

TREATMENT_RULES = {
    ("Apple", "Scab (General)", "General Disease"): [
        "Apply protective fungicides such as captan or dodine",
        "Prune to improve air circulation",
        "Remove and destroy fallen leaves to reduce overwintering spores"
    ],
    ("Apple", "Scab (Serious)", "Serious Disease"): [
        "Immediate application of systemic fungicides (e.g., myclobutanil)",
        "Remove and destroy severely infected trees",
        "Implement strict sanitation practices",
        "Consider resistant varieties for future plantings"
    ],
    ("Apple", "Cedar Apple Rust (General)", "General Disease"): [
        "Apply sulfur or copper-based fungicides",
        "Remove nearby juniper hosts if possible",
        "Monitor weather conditions for infection periods"
    ],
    ("Apple", "Cedar Apple Rust (Serious)", "Serious Disease"): [
        "Aggressive fungicide program with multiple applications",
        "Complete removal of nearby alternate hosts",
        "Consider planting resistant cultivars for future plantings"
    ],
    ("Cherry", "Powdery Mildew (General)", "General Disease"): [
        "Apply sulfur-based fungicides",
        "Prune to improve air circulation",
        "Avoid overhead irrigation"
    ],
    ("Cherry", "Powdery Mildew (Serious)", "Serious Disease"): [
        "Immediate application of systemic fungicides (e.g., myclobutanil)",
        "Remove and destroy severely infected branches",
        "Implement strict sanitation practices"
    ],
    ("Tomato", "Early Blight (General)", "General Disease"): [
        "Apply copper-based fungicides",
        "Remove lower infected leaves",
        "Rotate crops to non-solanaceous plants next season"
    ],
    ("Tomato", "Early Blight (Serious)", "Serious Disease"): [
        "Aggressive fungicide program with chlorothalonil",
        "Remove and destroy severely infected plants",
        "Implement 3-year crop rotation",
        "Consider resistant varieties for future plantings"
    ],
    ("Tomato", "Late Blight (General)", "General Disease"): [
        "Immediate application of mancozeb or chlorothalonil",
        "Remove infected leaves promptly",
        "Avoid working in fields when wet"
    ],
    ("Tomato", "Late Blight (Serious)", "Serious Disease"): [
        "Emergency fungicide applications every 5-7 days",
        "Complete removal of infected plants",
        "Implement strict field sanitation",
        "Consider resistant varieties for future plantings"
    ],
    (None, None, None): [
        "Consult local agricultural extension services",
        "Conduct laboratory testing for precise diagnosis",
        "Implement general cultural practices to improve plant health"
    ]
}


def generate_report_id():
    return f"RPT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_crop_from_label(label):
    disease_map = {
        0: "Apple", 1: "Apple", 2: "Apple", 3: "Apple", 4: "Apple", 5: "Apple",
        6: "Cherry", 7: "Cherry", 8: "Cherry",
        9: "Corn", 10: "Corn", 11: "Corn", 12: "Corn", 13: "Corn", 14: "Corn", 15: "Corn", 16: "Corn",
        17: "Grape", 18: "Grape", 19: "Grape", 20: "Grape", 21: "Grape", 22: "Grape", 23: "Grape",
        24: "Citrus", 25: "Citrus", 26: "Citrus",
        27: "Peach", 28: "Peach", 29: "Peach",
        30: "Pepper", 31: "Pepper", 32: "Pepper",
        33: "Potato", 34: "Potato", 35: "Potato", 36: "Potato", 37: "Potato",
        38: "Strawberry", 39: "Strawberry", 40: "Strawberry",
        41: "Tomato", 42: "Tomato", 43: "Tomato", 44: "Tomato", 45: "Tomato",
        46: "Tomato", 47: "Tomato", 48: "Tomato", 49: "Tomato", 50: "Tomato",
        51: "Tomato", 52: "Tomato", 53: "Tomato", 54: "Tomato", 55: "Tomato",
        56: "Tomato", 57: "Tomato", 58: "Tomato", 59: "Tomato", 60: "Tomato"
    }
    return disease_map.get(label, "Unknown Crop")


def get_disease_name(label):
    disease_names = {
        0: "Healthy", 1: "Scab (General)", 2: "Scab (Serious)", 3: "Frogeye Spot",
        4: "Cedar Apple Rust (General)", 5: "Cedar Apple Rust (Serious)",
        6: "Healthy", 7: "Powdery Mildew (General)", 8: "Powdery Mildew (Serious)",
        9: "Healthy", 10: "Cercospora Leaf Spot (General)", 11: "Cercospora Leaf Spot (Serious)",
        12: "Corn Rust (General)", 13: "Corn Rust (Serious)", 14: "Corn Curvularia Leaf Spot (General)",
        15: "Corn Curvularia Leaf Spot (Serious)", 16: "Maize Dwarf Mosaic Virus",
        41: "Healthy", 42: "Powdery Mildew (General)", 43: "Powdery Mildew (Serious)",
        44: "Bacterial Spot (General)", 45: "Bacterial Spot (Serious)",
        46: "Early Blight (General)", 47: "Early Blight (Serious)",
        48: "Late Blight (General)", 49: "Late Blight (Serious)",
        50: "Leaf Mold (General)", 51: "Leaf Mold (Serious)",
        52: "Septoria Leaf Spot (General)", 53: "Septoria Leaf Spot (Serious)",
        54: "Leaf Spot (General)", 55: "Leaf Spot (Serious)",
        56: "Spider Mites (General)", 57: "Spider Mites (Serious)",
        58: "Yellow Leaf Curl Virus", 59: "Yellow Leaf Curl Virus (Serious)",
        60: "Mosaic Virus"
    }
    return disease_names.get(label, "Unknown Disease")


def generate_severity_description(sev_pred, sev_conf):
    severity = SEVERITY_LEVELS[sev_pred]
    conf_level = "high" if sev_conf > 0.8 else "moderate" if sev_conf > 0.6 else "low"
    descriptions = {
        "Healthy": f"The plant shows no visible signs of disease with {conf_level} confidence. Continue regular monitoring and good agricultural practices.",
        "General Disease": f"The plant is showing early to moderate signs of disease with {conf_level} confidence. Immediate intervention can prevent significant crop loss.",
        "Serious Disease": f"The plant is severely affected with {conf_level} confidence. This represents a critical situation requiring immediate action to prevent spread to other plants and significant yield loss."
    }
    return descriptions.get(severity, "Severity assessment could not be determined.")


def format_list(items):
    return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])
