# 2025 数维杯 -- 农作物病害智能诊断系统

## 项目简介

基于深度学习的农作物病害多任务诊断系统，涵盖 10 种作物、61 类病害的分类识别、严重程度评估、可解释性分析（Grad-CAM）及自动化诊断报告生成。

## 任务概述

| 任务 | 描述 | 模型 | 类别数 |
|------|------|------|--------|
| Task 1 | 病害类型分类 | ResNet50 | 61 |
| Task 2 | 少样本病害分类 | SE-ResNet18 | 61 |
| Task 3 | 严重程度分类 | ResNet50 (迁移学习) | 3 |
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

## 模型架构

### Task 1: ResNet50
- 预训练 ImageNet 权重，替换最后一层为 61 类输出
- AdamW 优化器，AMP 混合精度训练

### Task 2: SE-ResNet18
- 在 ResNet18 的 layer3/layer4 添加 SE 注意力模块
- 少样本学习 (每类最多 10 张)，MixUp 增强，Label Smoothing

### Task 3: ResNet50 (迁移学习)
- 从 Task1 checkpoint 加载 backbone，替换为 3 类输出
- 健康 / 一般病害 / 严重病害

### Task 4: Multi-task ResNet50
- 共享 ResNet50 backbone + 两个分类头
- 加权多任务损失: `L = 1.0 * L_type + 1.2 * L_severity`
- 集成 Grad-CAM 可解释性分析

## 团队信息

队伍编号: BSHUWEICUP2533387
