from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class AgriDiseaseDataset(Dataset):
    def __init__(self, list_file, transform=None):
        self.samples = []
        self.transform = transform

        list_file = Path(list_file)
        root_dir = list_file.parent.parent

        with list_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path_str, label_str = line.split()
                path_str = path_str.replace("\\", "/")
                img_path = root_dir / path_str
                label = int(label_str)
                self.samples.append((img_path, label))

        print(f"[AgriDiseaseDataset] Loaded {len(self.samples)} samples from {list_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
