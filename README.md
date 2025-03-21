# GANForge - 基于GAN的图像生成平台

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Flet](https://img.shields.io/badge/GUI-Flet-green)

一个基于生成对抗网络（GAN）的图像生成工具，提供图形化界面方便用户训练模型和生成图像。

## 功能特性

- 🖼️ 支持灰度/彩色图像生成
- 🧠 可视化模型训练过程
- ⚙️ 灵活的参数配置
  - 自定义输入图像尺寸
  - 调整训练周期（Epoch）
  - 选择灰度/彩色模式
- 📁 模型管理功能
  - 导入/导出预训练模型（.pth）
  - 支持类别映射字典（.json）
- 🖥️ 友好的图形界面
  - 实时训练进度显示
  - 图像预览与保存功能
  - 跨平台支持（Windows/macOS/Linux）

## 安装指南

### 环境要求
- Python 3.8+
- CUDA 11.8（推荐，GPU加速）
- 至少4GB显存（GPU版本）

### 依赖安装
```bash
pip install -r requirements.txt
```

使用说明
快速启动
```bash
python ui.py
```
界面导航
模型训练

上传ZIP格式数据集（按类别分文件夹）

配置图像尺寸和颜色模式

导出训练好的生成器/判别器模型

内容生成

加载预训练模型和类别字典

实时生成指定类别的图像

支持PNG格式保存结果


项目结构
GANForge/

├── ui.py                # 主程序入口

├── GAN_model.py         # GAN模型定义与训练逻辑

├── model/               # 训练模型存储目录

├── generated_image/     # 生成结果保存目录

└── requirements.txt     # 依赖配置文件
