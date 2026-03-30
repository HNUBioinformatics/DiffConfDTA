# DiffConf-DTA: Drug-Target Binding Affinity Prediction

本项目提供了一个用于预测药物-靶点结合亲和力（Drug-Target Affinity, DTA）的深度学习模型。模型结合了**图卷积神经网络（GCN）**提取药物分子图特征，以及**注意力机制（Transformer/Conformer）**提取蛋白质序列特征，通过多模态特征融合进行精准的结合亲和力预测。

## ⚙️ 环境依赖 (Dependencies)

本项目主要在 **Python 3.10** 和 **CUDA 11.8** 环境下开发与测试。我们建议使用 `conda` 创建虚拟环境并安装所需的依赖库。

### 核心依赖库：

- `pytorch == 2.3.1` (CUDA 11.8)
- `torch-geometric == 2.3.1`
- `rdkit == 2022.03.2` 
- `transformers == 4.56.2` & `huggingface-hub == 0.35.0`
- `numpy == 1.26.4`
- `pandas == 2.3.2`
- `scikit-learn == 1.7.1`
- `scipy == 1.15.3`
- `biopython == 1.86`

### 环境配置步骤

你可以通过以下命令快速配置运行环境：

```bash
# 1. 创建 conda 虚拟环境
conda create -n dta_env python=3.10
conda activate dta_env

# 2. 安装 PyTorch (带 CUDA 11.8 支持)
conda install pytorch==2.3.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安装 PyTorch Geometric 及其他计算库
pip install torch_geometric==2.3.1
pip install numpy pandas scikit-learn scipy biopython transformers

# 4. 安装 RDKit
conda install -c conda-forge rdkit==2022.03.2

# 5. 启动
python main.py
```
