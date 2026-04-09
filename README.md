# DiffConf-DTA: Drug-Target Binding Affinity Prediction

This project provides a deep learning model for predicting drug-target affinity (DTA). The model combines a graph convolutional neural network (GCN) to extract drug molecule graph features and an attention mechanism (Transformer/Conformer) to extract protein sequence features, achieving accurate binding affinity prediction through multimodal feature fusion.

##  Dependencies

This project is primarily developed and tested in a Python 3.10 and CUDA 11.8 environment. We recommend using `conda` to create a virtual environment and install the necessary dependencies.

### Requirements：

- `pytorch == 2.3.1` (CUDA 11.8)
- `torch-geometric == 2.3.1`
- `rdkit == 2022.03.2` 
- `transformers == 4.56.2` & `huggingface-hub == 0.35.0`
- `numpy == 1.26.4`
- `pandas == 2.3.2`
- `scikit-learn == 1.7.1`
- `scipy == 1.15.3`
- `biopython == 1.86`

### Run

You can quickly configure the runtime environment using the following commands:

```bash
# 1. Create a conda virtual environment
conda create -n dta_env python=3.10
conda activate dta_env

# 2. Install PyTorch (with CUDA 11.8 support)
conda install pytorch==2.3.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install PyTorch Geometric and other computation libraries
pip install torch_geometric==2.3.1
pip install numpy pandas scikit-learn scipy biopython transformers

# 4. Install RDKitt
conda install -c conda-forge rdkit==2022.03.2

# 5. run
python main.py
```
