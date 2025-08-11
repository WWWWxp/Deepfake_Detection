# 深度伪造检测

基于最先进模型（包括AASIST和LCNN架构）的音频深度伪造检测深度学习框架。

## 功能特性

- 支持多种模型架构（AASIST、LCNN）
- 支持多GPU和单GPU训练
- 全面的评估指标
- 音频预处理和数据增强
- 预训练模型检查点

## 环境配置

### 前置要求
- Python 3.10+
- 支持CUDA的GPU（推荐）
- Conda包管理器

### 安装步骤

1. 创建并激活conda环境：
```bash
conda create -n deepfake python=3.10
conda activate deepfake
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

或者，您可以使用提供的conda环境文件：
```bash
conda env create -f environment.yml
conda activate deepfake
```

## 数据集准备

### 数据格式
按照以下结构准备您的数据集：

```
data/
├── train.txt
├── test.txt
└── audio_files/
    ├── file1.wav
    ├── file2.wav
    └── ...
```

### 标签文件格式
创建标签文件（train.txt、test.txt），格式如下：
```
file_id,file_path,label
0069156,aigc_speech_detection_tasks_part1/0069156.wav,Bonafide
0069157,aigc_speech_detection_tasks_part1/0069157.wav,Spoof
```

其中：
- `file_id`：音频文件的唯一标识符
- `file_path`：音频文件的相对路径
- `label`：标签，"Bonafide"（真实）或"Spoof"（伪造）

## 训练

### 多GPU训练
使用多个GPU进行训练：
```bash
bash train.sh
```

### 单GPU训练
使用单个GPU进行训练：
```bash
bash run_single.sh
```

### 自定义训练
您也可以使用自定义参数直接运行训练：
```bash
python train.py --config_path configs/your_config.yaml
```

## 推理

### 批量推理
对数据集运行推理：
```bash
python inference.py --model_path path/to/model.pth --data_path path/to/test_data
```

### 单文件推理
使用提供的shell脚本进行快速推理：
```bash
bash infer.sh path/to/audio_file.wav
```

## 模型架构

本项目支持多种架构：
- **AASIST**：使用集成频谱-时间图注意力的音频反欺骗技术
- **LCNN**：轻量级卷积神经网络

## 项目结构

```
├── module/                 # 模型架构
│   ├── AASIST.py          # AASIST模型实现
│   └── LCNN.py            # LCNN模型实现
├── scripts/               # 实用工具脚本
├── ckpts/                 # 预训练检查点
├── exports/               # 训练输出
├── dataset.py             # 数据集加载工具
├── model.py               # 模型包装器
├── learner.py             # 训练逻辑
├── inference.py           # 推理工具
├── metrics.py             # 评估指标
└── params.py              # 配置参数
```

## 结果

训练日志和模型检查点保存在`exports/`目录中。您可以使用提供的指标监控训练进度并评估模型性能。

## 贡献

欢迎提交问题和拉取请求来改进这个项目。

## 许可证

本项目采用MIT许可证 - 详情请参阅LICENSE文件。
