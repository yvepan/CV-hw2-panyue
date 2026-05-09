# 计算机视觉 HW2

本仓库包含计算机视觉 HW2 的代码、配置文件、实验报告和复现实验脚本。

作业包含三个任务：

- **任务 1：** 基于 ImageNet 预训练 CNN/Transformer 的 Oxford-IIIT Pet 宠物分类。
- **任务 2：** 基于 YOLOv8 的 VisDrone 目标检测，以及交通视频多目标跟踪。
- **任务 3：** 从零实现 U-Net，并完成宠物图像分割实验。

最终提交报告为 PDF 格式，同时保留 Markdown 源文件便于查看：

```text
report/final_report.pdf
report/final_report.md
```

相关链接：

- GitHub 仓库：<https://github.com/yvepan/CV-hw2-panyue>
- 模型权重下载：<https://drive.google.com/file/d/19YITjP_QeCgeC1rVTKM4qRmJJ8zOlmK4/view?usp=drive_link>

## 目录结构

```text
configs/                 三个任务的训练配置文件
scripts/                 数据准备、训练、跟踪、SwanLab 导入和快速测试脚本
src/hw2_cv/              核心源码
report/final_report.pdf   最终实验报告 PDF
report/final_report.md    最终实验报告 Markdown 源文件
report/assets/           报告中引用的图片
requirements.txt         Python 依赖
pyproject.toml           项目元信息
HW2_计算机视觉.pdf        作业原始说明文件
```

以下大文件或本地生成内容不纳入 GitHub，需要通过数据集下载、训练过程或网盘权重补齐：

```text
datasets/                本地数据集
results/                 本地 checkpoint、指标文件、跟踪结果和生成视频
runs/                    YOLOv8 训练输出
swanlog*/                本地 SwanLab 日志
*.pt, *.pth, *.onnx      模型权重
*.mp4, *.avi             视频文件
*.zip, *.rar, *.tar.gz   压缩包文件
```

## 环境要求

推荐环境：

- Python >= 3.10
- PyTorch >= 2.1
- 如需重新训练，建议使用支持 CUDA 的 PyTorch
- 如果只运行快速测试、检查代码结构或把已有指标导入 SwanLab，CPU 环境也可以完成

安装依赖：

```bash
python -m pip install -r requirements.txt
```

可选：以 editable 模式安装本项目：

```bash
python -m pip install -e .
```

如果需要使用 SwanLab 本地看板，建议安装 dashboard 额外依赖：

```bash
python -m pip install "swanlab[dashboard]"
```

主要依赖包括：

- `torch`、`torchvision`
- `timm`
- `ultralytics`
- `opencv-python`
- `matplotlib`
- `swanlab`

## 数据集、视频与模型权重

数据集、测试视频和模型权重体积较大，默认不通过 Git 管理。

复现实验时需要自行准备以下文件或目录：

| 内容 | 复现时的建议放置路径 | 说明 |
| --- | --- | --- |
| Oxford-IIIT Pet 数据集 | `datasets/oxford_pet/` | 任务 1 和任务 3 使用 |
| VisDrone 原始数据集 | `datasets/VisDrone/` | 任务 2 原始检测数据 |
| VisDrone YOLO 格式数据集 | `datasets/visdrone_yolo/` | 可由 `scripts/prepare_visdrone.py` 生成 |
| 任务 2 原始测试视频 | `180386-864121573_medium.mp4` | GitHub 仓库不包含视频文件 |
| 下载后的权重压缩包 | 任意本地目录 | 下载链接见 README 开头 |

权重压缩包内容：

```text
hw2_weights/
  task1/
    resnet18_imagenet_best.pt
    resnet18_random_init_best.pt
    resnet18_cbam_best.pt
    swin_t_imagenet_best.pt
  task2/
    yolov8n_visdrone_best.pt
  task3/
    unet_ce_best.pt
    unet_dice_best.pt
    unet_ce_dice_best.pt
```

如需直接运行本 README 中的跟踪命令，请将 `task2/yolov8n_visdrone_best.pt` 复制到：

```text
runs/detect/results/task2/visdrone_yolov8/weights/best.pt
```

如需按本项目脚本默认路径继续评估或复现实验，可将压缩包中的权重放到以下位置：

| 压缩包内文件 | 建议放置路径 |
| --- | --- |
| `task1/resnet18_imagenet_best.pt` | `results/task1/resnet18_imagenet/best.pt` |
| `task1/resnet18_random_init_best.pt` | `results/task1/resnet18_random_init/best.pt` |
| `task1/resnet18_cbam_best.pt` | `results/task1/resnet18_imagenet_cbam/best.pt` |
| `task1/swin_t_imagenet_best.pt` | `results/task1/swin_t_imagenet/best.pt` |
| `task2/yolov8n_visdrone_best.pt` | `runs/detect/results/task2/visdrone_yolov8/weights/best.pt` |
| `task3/unet_ce_best.pt` | `results/task3/unet_ce/best.pt` |
| `task3/unet_dice_best.pt` | `results/task3/unet_dice/best.pt` |
| `task3/unet_ce_dice_best.pt` | `results/task3/unet_ce_dice/best.pt` |

## 快速检查

不重新训练模型，只检查模型构建、loss、指标计算和跟踪计数逻辑：

```bash
python scripts/smoke_test.py
```

## 任务 1：宠物分类

训练全部分类实验：

```bash
python scripts/train_task1.py --config configs/task1_resnet18.yaml
python scripts/train_task1.py --config configs/task1_resnet18_random_init.yaml
python scripts/train_task1.py --config configs/task1_resnet18_cbam.yaml
python scripts/train_task1.py --config configs/task1_swin_t.yaml
```

结果摘要：

| 实验 | 最佳 Accuracy |
| --- | ---: |
| ResNet-18 ImageNet baseline | 0.8610 |
| ResNet-18 random init | 0.4050 |
| ResNet-18 + CBAM | 0.8673 |
| Swin-T ImageNet | 0.9302 |

## 任务 2：VisDrone 检测与跟踪

将 VisDrone 标注转换为 YOLO 格式：

```bash
python scripts/prepare_visdrone.py --root datasets/VisDrone --out datasets/visdrone_yolo
```

训练 YOLOv8：

```bash
python scripts/train_task2_yolo.py --config configs/task2_yolov8.yaml
```

准备 30 秒测试视频。该命令假设原始测试视频已放在项目根目录，文件名为 `180386-864121573_medium.mp4`：

```bash
python scripts/prepare_task2_clip.py \
  --src 180386-864121573_medium.mp4 \
  --dst results/task2/task2_test_30s_720p.mp4 \
  --duration 30 --fps 25 --width 1280
```

运行多目标跟踪与越线计数。该命令假设已将下载的 YOLO 权重放到 `--weights` 指定路径：

```bash
python scripts/track_task2.py \
  --weights runs/detect/results/task2/visdrone_yolov8/weights/best.pt \
  --video results/task2/task2_test_30s_720p.mp4 \
  --out results/task2/tracking_final \
  --line 0.05 0.64 0.95 0.64 \
  --analyze-start 250 --analyze-len 4
```

结果摘要：

| 指标 | 结果 |
| --- | ---: |
| YOLOv8 best mAP50 | 0.2993 |
| YOLOv8 best mAP50-95 | 0.1687 |
| 最终视频时长 | 30.00 s |
| 跟踪记录数 | 25253 |
| 越线计数 | 38 |
| 可能 ID 跳变 | 2 |

## 任务 3：U-Net 分割

训练全部分割实验：

```bash
python scripts/train_task3.py --config configs/task3_unet_ce.yaml
python scripts/train_task3.py --config configs/task3_unet_dice.yaml
python scripts/train_task3.py --config configs/task3_unet_ce_dice.yaml
```

结果摘要：

| 损失函数 | 最佳 mIoU |
| --- | ---: |
| Cross-Entropy | 0.7615 |
| Dice | 0.7720 |
| Cross-Entropy + Dice | 0.7739 |

## SwanLab 可视化

如果本地已有训练产生的 CSV 指标，可以直接导入 SwanLab，无需重新训练。注意：GitHub 仓库不包含完整 `results/` 和 `runs/` 目录，因此全新环境需要先完成训练或自行准备对应日志文件。

```bash
python scripts/export_to_swanlab.py --mode local --project cv-hw2 --logdir swanlog
python -m swanlab watch swanlog
```

如果希望三个任务分别打开独立看板：

```bash
python scripts/export_to_swanlab.py --mode local --project cv-hw2-task1 --logdir swanlog_task1 --tasks task1
python scripts/export_to_swanlab.py --mode local --project cv-hw2-task2 --logdir swanlog_task2 --tasks task2
python scripts/export_to_swanlab.py --mode local --project cv-hw2-task3 --logdir swanlog_task3 --tasks task3
```

然后根据终端输出的本地地址打开 SwanLab：

```bash
python -m swanlab watch swanlog_task1
python -m swanlab watch swanlog_task2
python -m swanlab watch swanlog_task3
```

如果需要上传到 SwanLab 云端：

```bash
python -m swanlab login
python scripts/export_to_swanlab.py --mode cloud --project cv-hw2 --logdir swanlog
```

只检查将导入哪些实验、不写入日志：

```bash
python scripts/export_to_swanlab.py --dry-run
```

## 报告图片

最终报告中的训练曲线使用 SwanLab 看板截图：

```text
report/assets/task1_val_accuracy.png
report/assets/task2_yolo_results.png
report/assets/task3_curves.png
```

跟踪帧截图和补充 PR 曲线可以由已有实验结果重新生成：

```bash
python report/make_assets.py
```

请保持 `report/assets/` 下图片路径不变，否则 `report/final_report.md` 中的图片引用会失效。
