# 2025 数维杯 -- 农作物病害智能诊断系统

## 项目简介

基于深度学习的农作物病害多任务诊断系统，涵盖 10 种作物、61 类病害的分类识别、严重程度评估、可解释性分析（Grad-CAM）及自动化诊断报告生成。

## 任务概述


| 任务   | 描述           | 模型                | 类别数 |
| ------ | -------------- | ------------------- | ------ |
| Task 1 | 病害类型分类   | ResNet50            | 61     |
| Task 2 | 少样本病害分类 | SE-ResNet18         | 61     |
| Task 3 | 严重程度分类   | ResNet50 (迁移学习) | 3      |
| Task 4 | 多任务联合诊断 | Multi-task ResNet50 | 61 + 3 |

## 目录结构

```
2025数维杯/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models/                    # 模型定义
│   │   ├── se_resnet.py           # SE 注意力模块 (SEBlock, SEBasicBlock)
│   │   ├── task1_resnet.py        # Task1 ResNet50 构建
│   │   ├── task2_se_resnet.py     # Task2 SE-ResNet18 构建
│   │   └── task4_resnet.py        # Task4 多任务模型
│   ├── datasets/                  # 数据集与映射
│   │   ├── agri_disease.py        # 基础数据集类
│   │   ├── severity_mapping.py    # 61类→3级严重度映射
│   │   ├── transforms.py          # 数据增强
│   │   └── task4_dataset.py       # 多任务数据集
│   ├── training/                  # 训练脚本
│   │   ├── train_task1.py         # Task1 训练
│   │   ├── train_task2.py         # Task2 少样本训练
│   │   ├── train_task3.py         # Task3 训练
│   │   └── train_task4.py         # Task4 多任务训练
│   ├── inference/                 # 推理与报告生成
│   │   ├── infer_task4.py         # Task4 批量推理
│   │   ├── report_generator.py    # 诊断报告生成器
│   │   ├── feature_index.py       # 病害特征与治疗规则
│   │   └── generate_explainable_reports.py  # 可解释性报告
│   ├── visualization/             # 可视化
│   │   ├── vis_task1.py           # Task1 混淆矩阵 + Grad-CAM
│   │   ├── vis_task2.py           # Task2 混淆矩阵 + Grad-CAM
│   │   ├── vis_task3.py           # Task3 混淆矩阵 + Grad-CAM
│   │   ├── vis_task4.py           # Task4 相关性分析
│   │   └── vis_overview.py        # 综合可视化概览
│   └── utils/                     # 共享工具
│       └── gradcam.py             # 统一 Grad-CAM 实现
├── tools/                         # 辅助工具
│   ├── plot_task1_log.py          # 训练曲线绘制
│   └── stat_raw_severity.py       # 数据集统计
└── outputs/                       # 生成的输出
    └── task4/
        ├── explainable_reports/   # 可解释性诊断报告
        ├── inference_results/     # 推理结果 JSON
        └── sample_reports/        # 样本报告
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据集

数据集目录结构：

```
data/
├── AgriculturalDisease_trainingset/
│   ├── images/
│   └── train_list.txt
└── AgriculturalDisease_validationset/
    ├── images/
    └── ttest_list.txt
```

每行格式: `<相对图片路径> <类别标签>`

覆盖 10 种作物：Apple, Cherry, Corn, Grape, Citrus, Peach, Pepper, Potato, Strawberry, Tomato

## 使用方法

### 训练

```bash
# Task 1: 病害类型分类
python -m src.training.train_task1

# Task 2: 少样本分类
python -m src.training.train_task2

# Task 3: 严重程度分类 (需要 Task1 权重)
python -m src.training.train_task3

# Task 4: 多任务联合诊断
python -m src.training.train_task4
```

### 推理

```bash
# 批量推理 + 生成报告
python -m src.inference.infer_task4

# 生成可解释性报告
python -m src.inference.generate_explainable_reports
```

### 可视化

```bash
python -m src.visualization.vis_task1
python -m src.visualization.vis_task2
python -m src.visualization.vis_task3
python -m src.visualization.vis_task4
python -m src.visualization.vis_overview
```

## 核心技术方案

### 1. 基础架构选型


| 技术                            | 应用任务   | 说明                                                                                              |
| ------------------------------- | ---------- | ------------------------------------------------------------------------------------------------- |
| **ResNet50 + ImageNet 预训练**  | Task 1/3/4 | 利用 ImageNet 学到的通用视觉特征作为起点，大幅减少训练数据需求，加速收敛                          |
| **SE 注意力模块 (SE-ResNet18)** | Task 2     | Squeeze-and-Excitation 机制自适应加权通道特征，提升少样本场景下的判别力                           |
| **多任务学习架构**              | Task 4     | 共享 ResNet50 backbone + 双分类头（病害类型 61 类 + 严重程度 3 类），利用任务间相关性提升泛化能力 |

### 2. 训练策略


| 技术                        | 应用任务  | 说明                                                                     |
| --------------------------- | --------- | ------------------------------------------------------------------------ |
| **迁移学习**              | Task 1→3 | Task1 学到的病害特征迁移到严重度评估任务；ImageNet→SE-ResNet18 跨域迁移 |
| **层冻结**                  | Task 2    | 仅训练 layer3/layer4/fc，冻结底层参数，减少参数量防止少样本过拟合        |
| **AMP 混合精度训练**        | Task 1    | FP16 前向传播 + FP32 梯度更新，加速训练并减少显存占用                    |
| **MixUp 数据增强**          | Task 2    | α=0.2 的 MixUp 生成虚拟样本，有效扩充少样本训练集                       |
| **Label Smoothing**         | Task 2    | smoothing=0.1 的软标签，防止模型对少样本类别过度自信                     |
| **Cosine Annealing 学习率** | Task 2    | 余弦退火平滑调整学习率，避免学习率突变导致的训练不稳定                   |
| **Early Stopping**          | Task 2    | patience=10 的早停机制，自动停止训练防止过拟合                           |
| **AdamW 优化器**            | 全部      | 带权重衰减的自适应优化，所有任务统一使用                                 |

### 3. 数据处理


| 技术                | 说明                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **数据增强流水线**  | RandomResizedCrop(224) + RandomHorizontalFlip + ColorJitter，提升模型对光照/角度/尺度的鲁棒性 |
| **ImageNet 标准化** | 使用 ImageNet 均值 [0.485, 0.456, 0.406] 和标准差 [0.229, 0.224, 0.225] 归一化                |
| **严重度标签派生**  | 从 61 类病害标签通过规则映射到 3 级：健康(0) / 一般病害(1) / 严重病害(2)                      |
| **少样本子集构建**  | Task 2 每类最多 10 张样本，模拟真实农业生产中的少样本诊断场景                                 |

### 4. 模型优化


| 技术                  | 说明                                                                                |
| --------------------- | ----------------------------------------------------------------------------------- |
| **加权多任务损失**    | `L = 1.0 × L_type + 1.2 × L_severity`，严重度分支权重更高，平衡两个任务的学习速度 |
| **Dropout(0.5)**      | 两个分类头均使用 Dropout 正则化，防止过拟合                                         |
| **Macro-F1 模型选择** | Task 4 使用宏平均 F1 而非准确率选模型，平衡各类别表现，避免多数类主导               |

### 5. 可解释性与应用


| 技术                  | 说明                                                                            |
| --------------------- | ------------------------------------------------------------------------------- |
| **Grad-CAM 双热力图** | 同时可视化病害类型和严重程度两个任务的决策区域，定位模型关注的病灶区域          |
| **自动化诊断报告**    | 基于推理结果 + 知识库规则，自动生成包含病害名称、严重程度、治疗建议的诊断报告   |
| **知识库驱动**        | `feature_index.py` 存储 61 类病害的特征描述和针对性治疗方案，实现知识增强的诊断 |
