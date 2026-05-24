import os
from collections import Counter
from pathlib import Path

# 61类 --> 原始严重度映射表（Healthy/General/Serious/Unknown）
ID2RAW_SEVERITY = {
    0: "Healthy", 1: "General", 2: "Serious", 3: "Unknown",
    4: "General", 5: "Serious",
    6: "Healthy", 7: "General", 8: "Serious",
    9: "Healthy", 10: "General", 11: "Serious",
    12: "General", 13: "Serious", 14: "General", 15: "Serious",
    16: "Unknown",
    17: "Healthy", 18: "General", 19: "Serious",
    20: "General", 21: "Serious", 22: "General", 23: "Serious",
    24: "Healthy", 25: "General", 26: "Serious",
    27: "Healthy", 28: "General", 29: "Serious",
    30: "Healthy", 31: "General", 32: "Serious",
    33: "Healthy", 34: "General", 35: "Serious",
    36: "General", 37: "Serious",
    38: "Healthy", 39: "General", 40: "Serious",
    41: "Healthy", 42: "General", 43: "Serious",
    44: "General", 45: "Serious", 46: "General", 47: "Serious",
    48: "General", 49: "Serious", 50: "General", 51: "Serious",
    52: "General", 53: "Serious", 54: "General", 55: "Serious",
    56: "General", 57: "Serious", 58: "General", 59: "Serious",
    60: "Unknown",
}

def read_list_file(list_path):
    labels = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label_str = line.split()
            labels.append(int(label_str))
    return labels

def main():

    project_root = Path(__file__).resolve().parents[1]


    train_list = project_root / "data/AgriculturalDisease_trainingset/train_list.txt"
    val_list   = project_root / "data/AgriculturalDisease_validationset/ttest_list.txt"

    print(f"[Info] Train list: {train_list}")
    print(f"[Info] Val list:   {val_list}")

    train_labels = read_list_file(train_list)
    val_labels   = read_list_file(val_list)

    all_labels = train_labels + val_labels

    cnt_class = Counter(all_labels)
    cnt_raw_sev = Counter()

    for cls_id, n in cnt_class.items():
        raw = ID2RAW_SEVERITY.get(cls_id, "Unknown")
        cnt_raw_sev[raw] += n

    print("\n=== Raw severity distribution ===")
    total = sum(cnt_raw_sev.values())
    for sev in ["Healthy", "General", "Serious", "Unknown"]:
        n = cnt_raw_sev.get(sev, 0)
        print(f"{sev:8s}: {n:5d}  ({n/total:.3%})")

    print("\n=== Class-level counts ===")
    for i in range(61):
        print(f"Class {i:2d}: {cnt_class.get(i,0):5d}, raw={ID2RAW_SEVERITY[i]}")

if __name__ == "__main__":
    main()
