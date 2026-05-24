from src.datasets.agri_disease import AgriDiseaseDataset
from src.datasets.severity_mapping import get_severity3_id


class MultiTaskDataset(AgriDiseaseDataset):
    def __getitem__(self, idx):
        img, disease_class = super().__getitem__(idx)
        severity_id = get_severity3_id(disease_class)
        return img, disease_class, severity_id

    def get_crop_disease_name(self, class_id):
        id_to_name = {
            0: "Apple Healthy",
            1: "Apple Scab (General)",
            2: "Apple Scab (Serious)",
            3: "Apple Frogeye Spot (General)",
            4: "Cedar Apple Rust (General)",
            5: "Cedar Apple Rust (Serious)",
            6: "Cherry Healthy",
            7: "Cherry Powdery Mildew (General)",
            8: "Cherry Powdery Mildew (Serious)",
            9: "Corn Healthy",
            10: "Corn Cercospora Leaf Spot (General)",
            11: "Corn Cercospora Leaf Spot (Serious)",
            12: "Corn Rust (General)",
            13: "Corn Rust (Serious)",
            14: "Corn Curvularia Leaf Spot (General)",
            15: "Corn Curvularia Leaf Spot (Serious)",
            16: "Maize Dwarf Mosaic Virus (General)",
            17: "Grape Healthy",
            18: "Grape Black Rot (General)",
            19: "Grape Black Rot (Serious)",
            20: "Grape Black Measles (General)",
            21: "Grape Black Measles (Serious)",
            22: "Grape Leaf Blight (General)",
            23: "Grape Leaf Blight (Serious)",
            24: "Citrus Healthy",
            25: "Citrus Greening (General)",
            26: "Citrus Greening (Serious)",
            27: "Peach Healthy",
            28: "Peach Bacterial Spot (General)",
            29: "Peach Bacterial Spot (Serious)",
            30: "Pepper Healthy",
            31: "Pepper Scab (General)",
            32: "Pepper Scab (Serious)",
            33: "Potato Healthy",
            34: "Potato Early Blight (General)",
            35: "Potato Early Blight (Serious)",
            36: "Potato Late Blight (General)",
            37: "Potato Late Blight (Serious)",
            38: "Strawberry Healthy",
            39: "Strawberry Scorch (General)",
            40: "Strawberry Scorch (Serious)",
            41: "Tomato Healthy",
            42: "Tomato Powdery Mildew (General)",
            43: "Tomato Powdery Mildew (Serious)",
            44: "Tomato Bacterial Spot (General)",
            45: "Tomato Bacterial Spot (Serious)",
            46: "Tomato Early Blight (General)",
            47: "Tomato Early Blight (Serious)",
            48: "Tomato Late Blight (General)",
            49: "Tomato Late Blight (Serious)",
            50: "Tomato Leaf Mold (General)",
            51: "Tomato Leaf Mold (Serious)",
            52: "Tomato Septoria Leaf Spot (General)",
            53: "Tomato Septoria Leaf Spot (Serious)",
            54: "Tomato Leaf Spot (General)",
            55: "Tomato Leaf Spot (Serious)",
            56: "Tomato Spider Mites (General)",
            57: "Tomato Spider Mites (Serious)",
            58: "Tomato Yellow Leaf Curl Virus (General)",
            59: "Tomato Yellow Leaf Curl Virus (Serious)",
            60: "Tomato Mosaic Virus (General)"
        }
        return id_to_name.get(class_id, "Unknown Crop Disease")
